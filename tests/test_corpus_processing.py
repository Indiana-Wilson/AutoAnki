import ast
import builtins
import errno
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from corpus_pipeline.audit import audit_build, audit_snapshot
from corpus_pipeline.chunks import (
    chunk_records,
    chunk_text,
    make_chunks,
)
from corpus_pipeline.contexts import (
    assemble_sections,
    build_contexts,
    with_section_titles,
)
from corpus_pipeline.models import (
    CORPUS_PROCESSING_VERSION,
    BuildConfig,
    CorpusBuild,
    CorpusSnapshot,
    CorpusValidationError,
    SourcePage,
    TextSection,
    TokenSpan,
    TokenizerIdentity,
    UniqueWord,
)
from corpus_pipeline.processing import build_vocabulary, is_han_word
from corpus_pipeline.storage import (
    build_id,
    read_build,
    read_snapshot,
    read_vocabulary_view,
    snapshot_id,
    write_build,
    write_snapshot,
)
from corpus_pipeline.tokenizers import (
    CKIP_BATCH_SCHEDULER_VERSION,
    CKIP_DECODER_VERSION,
    CKIP_REFINEMENT_VERSION,
    CKIP_UNITIZER_VERSION,
    CharacterTokenizer,
    CkipHanTokenizer,
    CorpusTokenizerUnavailableError,
    split_model_units,
)


class FakeCharacterTokenizer:
    """Small deterministic tokenizer with no model or external dependency."""

    @property
    def identity(self):
        return TokenizerIdentity(
            backend="test-character",
            backend_version="1.0",
            model="test-code-points",
            model_revision="fixture-1",
            options=(("whitespace", "omit"),))

    def tokenize(self, text):
        return tuple(
            TokenSpan(
                surface=character,
                start_offset=index,
                end_offset=index + 1,
                confidence=0.875)
            for index, character in enumerate(text)
            if not character.isspace())


class FixedTokenizer:
    def __init__(self, spans):
        self._spans = tuple(spans)

    @property
    def identity(self):
        return TokenizerIdentity(
            backend="test-fixed",
            backend_version="1",
            model="fixed-spans",
            model_revision="1")

    def tokenize(self, _text):
        return self._spans


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_page(
        *,
        page_key="page-001",
        order=1,
        raw_wikitext="== 第一章 ==\n道可道。",
        retrieved_at="2026-07-24T00:00:00+00:00"):
    return SourcePage(
        page_key=page_key,
        order=order,
        title="測試來源",
        url=f"https://example.invalid/wiki/{page_key}",
        revision_id=123456,
        revision_timestamp="2026-07-23T00:00:00Z",
        revision_sha1="0123456789abcdef",
        retrieved_at=retrieved_at,
        raw_wikitext=raw_wikitext,
        raw_sha256=_sha256(raw_wikitext))


def make_snapshot(
        section_texts=("道可道。名可名！\n\n無名，名無。", "名德。"),
        *,
        raw_wikitext="== 第一章 ==\n道可道。",
        spec_key="fixture_work",
        cleaner_version="fixture-cleaner-v1"):
    page = make_page(raw_wikitext=raw_wikitext)
    sections = tuple(
        TextSection(
            section_id=f"section-{order:03d}",
            order=order,
            title=f"第 {order} 節",
            source_page_key=page.page_key,
            text=text)
        for order, text in enumerate(section_texts, start=1))
    aligned, canonical_text = assemble_sections(sections)
    return CorpusSnapshot(
        spec_key=spec_key,
        edition="Fixture edition",
        source_language_key="classical_chinese",
        pages=(page,),
        sections=aligned,
        canonical_text=canonical_text,
        cleaner_version=cleaner_version)


def make_build(
        section_texts=("道可道。名可名！\n\n無名，名無。", "名德。"),
        *,
        chunk_size=500):
    return build_vocabulary(
        make_snapshot(section_texts),
        FakeCharacterTokenizer(),
        BuildConfig(chunk_size=chunk_size))


def make_words(count):
    return tuple(
        UniqueWord(
            rank=rank,
            surface=f"詞{rank}",
            normalized=f"詞{rank}",
            first_occurrence_id=f"token:{rank:09d}",
            occurrence_count=1,
            section_id="section-001",
            start_offset=rank - 1,
            end_offset=rank,
            paragraph_id="section-001:paragraph:0001",
            sentence_id="section-001:sentence:00001")
        for rank in range(1, count + 1))


class ContextAssemblyTests(unittest.TestCase):
    def test_assemble_sections_normalizes_newlines_nfc_and_exact_offsets(self):
        sections = (
            TextSection(
                section_id="one",
                order=1,
                title="One",
                source_page_key="page-001",
                text="  e\u0301道\r\n德  "),
            TextSection(
                section_id="two",
                order=2,
                title="Two",
                source_page_key="page-001",
                text="\r\n無名\r\n"),
        )

        aligned, canonical = assemble_sections(sections)

        self.assertEqual(canonical, "é道\n德\n\n無名")
        self.assertEqual(
            tuple(
                (item.text, item.start_offset, item.end_offset)
                for item in aligned),
            (("é道\n德", 0, 4), ("無名", 6, 8)))
        for section in aligned:
            self.assertEqual(
                canonical[section.start_offset:section.end_offset],
                section.text)

    def test_assemble_sections_rejects_empty_duplicate_and_reordered_sections(
            self):
        base = TextSection(
            section_id="one",
            order=1,
            title="One",
            source_page_key="page-001",
            text="道")
        cases = (
            (replace(base, text=" \r\n "),),
            (base, replace(base, order=2, text="德")),
            (replace(base, order=2),),
        )
        for sections in cases:
            with self.subTest(sections=sections):
                with self.assertRaises(CorpusValidationError):
                    assemble_sections(sections)

    def test_contexts_have_exact_global_offsets_and_text(self):
        snapshot = make_snapshot()

        contexts = build_contexts(
            snapshot.canonical_text,
            snapshot.sections)

        self.assertEqual(
            tuple(
                (
                    context.context_id,
                    context.kind,
                    context.start_offset,
                    context.end_offset,
                    context.text,
                )
                for context in contexts),
            (
                (
                    "section-001:paragraph:0001",
                    "paragraph",
                    0,
                    8,
                    "道可道。名可名！",
                ),
                (
                    "section-001:sentence:00001",
                    "sentence",
                    0,
                    4,
                    "道可道。",
                ),
                (
                    "section-001:sentence:00002",
                    "sentence",
                    4,
                    8,
                    "名可名！",
                ),
                (
                    "section-001:paragraph:0002",
                    "paragraph",
                    10,
                    16,
                    "無名，名無。",
                ),
                (
                    "section-001:sentence:00003",
                    "sentence",
                    10,
                    16,
                    "無名，名無。",
                ),
                (
                    "section-002:paragraph:0001",
                    "paragraph",
                    18,
                    21,
                    "名德。",
                ),
                (
                    "section-002:sentence:00001",
                    "sentence",
                    18,
                    21,
                    "名德。",
                ),
            ))
        for context in contexts:
            self.assertEqual(
                snapshot.canonical_text[
                    context.start_offset:context.end_offset],
                context.text)

    def test_section_titles_can_be_included_and_removed_losslessly(self):
        snapshot = make_snapshot(("正文。", "次章。"))

        included = with_section_titles(snapshot, True)
        restored = with_section_titles(included, False)

        self.assertTrue(included.include_section_titles)
        self.assertEqual(
            included.sections[0].text,
            "第 1 節\n\n正文。")
        self.assertEqual(
            included.sections[1].text,
            "第 2 節\n\n次章。")
        self.assertEqual(restored, snapshot)
        self.assertIs(with_section_titles(included, True), included)


class TokenProcessingTests(unittest.TestCase):
    def test_build_config_rejects_unimplemented_or_ambiguous_values(self):
        invalid_options = (
            {"chunk_size": True},
            {"chunk_size": 2.5},
            {"chunk_size": 0},
            {"normalization": "NFKC"},
            {"context_policy": "sentences-only"},
            {"include_section_titles": "yes"},
        )
        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    BuildConfig(**options)

    def test_han_filter_accepts_han_marks_extensions_and_variation_selectors(
            self):
        accepted = (
            "道",
            "天地",
            "々",
            "〇",
            "\U00020000",
            "漢\ufe0f",
            "漢\U000e0100",
        )
        rejected = (
            "",
            "道。",
            "漢字A",
            "かな",
            "，",
            "\ufe0f",
            "道\u0301",
        )
        for text in accepted:
            with self.subTest(text=text):
                self.assertTrue(is_han_word(text))
        for text in rejected:
            with self.subTest(text=text):
                self.assertFalse(is_han_word(text))

    def test_vocabulary_offsets_contexts_order_counts_and_filtering(self):
        build = make_build()

        self.assertEqual(
            tuple(item.surface for item in build.occurrences),
            (
                "道",
                "可",
                "道",
                "名",
                "可",
                "名",
                "無",
                "名",
                "名",
                "無",
                "名",
                "德",
            ))
        self.assertEqual(
            tuple(
                (item.start_offset, item.end_offset)
                for item in build.occurrences),
            (
                (0, 1),
                (1, 2),
                (2, 3),
                (4, 5),
                (5, 6),
                (6, 7),
                (10, 11),
                (11, 12),
                (13, 14),
                (14, 15),
                (18, 19),
                (19, 20),
            ))
        self.assertEqual(
            tuple(item.occurrence_id for item in build.occurrences),
            tuple(
                f"token:{index:09d}"
                for index in range(1, 13)))
        self.assertEqual(
            tuple(item.confidence for item in build.occurrences),
            (0.875,) * 12)
        self.assertEqual(
            (
                build.occurrences[0].paragraph_id,
                build.occurrences[0].sentence_id,
                build.occurrences[3].sentence_id,
                build.occurrences[6].paragraph_id,
                build.occurrences[10].section_id,
            ),
            (
                "section-001:paragraph:0001",
                "section-001:sentence:00001",
                "section-001:sentence:00002",
                "section-001:paragraph:0002",
                "section-002",
            ))
        self.assertEqual(
            tuple(
                (
                    word.rank,
                    word.surface,
                    word.occurrence_count,
                    word.start_offset,
                    word.first_occurrence_id,
                )
                for word in build.unique_words),
            (
                (1, "道", 2, 0, "token:000000001"),
                (2, "可", 2, 1, "token:000000002"),
                (3, "名", 5, 4, "token:000000004"),
                (4, "無", 2, 10, "token:000000007"),
                (5, "德", 1, 19, "token:000000012"),
            ))
        self.assertTrue(audit_build(build))

    def test_nfc_is_canonical_before_tokenization_and_in_output(self):
        snapshot = make_snapshot(("e\u0301道",))
        self.assertEqual(snapshot.canonical_text, "é道")

        build = build_vocabulary(
            snapshot,
            FakeCharacterTokenizer(),
            BuildConfig())

        self.assertEqual(len(build.occurrences), 1)
        self.assertEqual(
            (
                build.occurrences[0].surface,
                build.occurrences[0].normalized,
                build.occurrences[0].start_offset,
                build.occurrences[0].end_offset,
            ),
            ("道", "道", 1, 2))

    def test_invalid_tokenizer_offsets_fail_closed(self):
        snapshot = make_snapshot(("道德",))
        invalid_span_sets = (
            (
                TokenSpan("道", 0, 1),
                TokenSpan("道", 0, 1),
            ),
            (TokenSpan("德", 0, 1),),
            (TokenSpan("道", 1, 1),),
            (TokenSpan("道", -1, 0),),
        )
        for spans in invalid_span_sets:
            with self.subTest(spans=spans):
                with self.assertRaises(CorpusValidationError):
                    build_vocabulary(
                        snapshot,
                        FixedTokenizer(spans),
                        BuildConfig())

    def test_mixed_tokenizer_spans_keep_each_maximal_han_run(self):
        snapshot = make_snapshot(("，容。夫",))
        build = build_vocabulary(
            snapshot,
            FixedTokenizer((
                TokenSpan("，容。夫", 0, 4, 0.75),
            )),
            BuildConfig())

        self.assertEqual(
            tuple(
                (
                    occurrence.surface,
                    occurrence.start_offset,
                    occurrence.end_offset,
                    occurrence.confidence,
                )
                for occurrence in build.occurrences),
            (
                ("容", 1, 2, 0.75),
                ("夫", 3, 4, 0.75),
            ))
        self.assertTrue(audit_build(build))

    def test_build_audit_requires_every_han_character_to_be_covered(self):
        snapshot = make_snapshot(("道容",))
        build = build_vocabulary(
            snapshot,
            FixedTokenizer((TokenSpan("道", 0, 1),)),
            BuildConfig())

        with self.assertRaisesRegex(
                CorpusValidationError,
                "Han/variation code point at offset 1"):
            audit_build(build)

        variation_snapshot = make_snapshot(("漢\ufe0f",))
        variation_build = build_vocabulary(
            variation_snapshot,
            FixedTokenizer((TokenSpan("漢", 0, 1),)),
            BuildConfig())
        with self.assertRaisesRegex(
                CorpusValidationError,
                "Han/variation code point at offset 1"):
            audit_build(variation_build)

    def test_section_offset_corruption_fails_before_tokenization(self):
        snapshot = make_snapshot(("道德",))
        broken_section = replace(
            snapshot.sections[0],
            start_offset=1,
            end_offset=3)
        broken = replace(snapshot, sections=(broken_section,))

        with self.assertRaisesRegex(
                CorpusValidationError,
                "Section offsets do not reproduce"):
            build_vocabulary(
                broken,
                FakeCharacterTokenizer(),
                BuildConfig())


class TokenizerUtilityTests(unittest.TestCase):
    @staticmethod
    def _decoder():
        class Config:
            id2label = {
                0: "B",
                1: "I",
            }

        class Model:
            config = Config()

        tokenizer = CkipHanTokenizer(
            "fixture/model",
            "fixture-revision")
        tokenizer._model = Model()
        return tokenizer

    def test_model_units_prefer_natural_boundary_without_dropping_text(self):
        text = ("甲" * 39) + "。" + ("乙" * 35)

        units = split_model_units(text, max_characters=64)

        self.assertEqual(
            tuple((offset, len(source)) for offset, source in units),
            ((0, 40), (40, 35)))
        self.assertEqual("".join(source for _, source in units), text)
        for (offset, source), next_unit in zip(
                units,
                units[1:],
                strict=False):
            self.assertEqual(offset + len(source), next_unit[0])

    def test_model_units_isolate_sentences_and_unterminated_paragraphs(self):
        text = (
            "甲乙。 \n"
            "丙丁！？』\n\n"
            "沒有句號的段落\n\n"
            "最後一句")

        units = split_model_units(text, max_characters=64)

        self.assertEqual(
            tuple(source for _, source in units),
            (
                "甲乙。 \n",
                "丙丁！？』\n\n",
                "沒有句號的段落\n\n",
                "最後一句",
            ))
        self.assertEqual("".join(source for _, source in units), text)
        for (offset, source), next_unit in zip(
                units,
                units[1:],
                strict=False):
            self.assertEqual(offset + len(source), next_unit[0])

    def test_ckip_adapter_validates_model_limits_and_records_batch_size(self):
        invalid_options = (
            {"batch_size": True},
            {"batch_size": 1.5},
            {"batch_size": 0},
            {"max_characters": True},
            {"max_characters": 31},
            {"max_characters": 510.5},
            {"max_characters": 511},
            {"device": "gpu"},
            {"device": "cuda:one"},
            {"device": "xpu:-1"},
        )
        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    CkipHanTokenizer(
                        "fixture/model",
                        "fixture-revision",
                        **options)

        tokenizer = CkipHanTokenizer(
            "fixture/model",
            "fixture-revision",
            batch_size=3,
            max_characters=510)
        self.assertEqual(
            dict(tokenizer.identity.options)["batch_size"],
            "3")
        self.assertEqual(
            dict(tokenizer.identity.options)["unitizer_version"],
            CKIP_UNITIZER_VERSION)
        self.assertEqual(
            dict(tokenizer.identity.options)["refinement_version"],
            CKIP_REFINEMENT_VERSION)
        self.assertEqual(
            dict(tokenizer.identity.options)["batch_scheduler_version"],
            CKIP_BATCH_SCHEDULER_VERSION)
        self.assertIn(
            "pytorch_version",
            dict(tokenizer.identity.options))
        self.assertIn(
            "tokenizers_version",
            dict(tokenizer.identity.options))
        self.assertIn(
            "python_version",
            dict(tokenizer.identity.options))
        self.assertEqual(
            dict(tokenizer.identity.options)["device"],
            "auto")

    def test_ckip_auto_device_prefers_nvidia_then_intel_then_mps(self):
        class Available:
            def __init__(self, value):
                self.value = value

            def is_available(self):
                return self.value

        class Backends:
            mps = Available(False)

        class Torch:
            cuda = Available(True)
            xpu = Available(True)
            backends = Backends()

        tokenizer = CkipHanTokenizer(
            "fixture/model",
            "fixture-revision")
        self.assertEqual(tokenizer._resolve_device(Torch()), "cuda")

        Torch.cuda = Available(False)
        self.assertEqual(tokenizer._resolve_device(Torch()), "xpu")

        Torch.xpu = Available(False)
        Torch.backends.mps = Available(True)
        self.assertEqual(tokenizer._resolve_device(Torch()), "mps")

        Torch.backends.mps = Available(False)
        self.assertEqual(tokenizer._resolve_device(Torch()), "cpu")

    def test_model_units_use_hard_boundaries_and_cover_edge_cases(self):
        text = "甲" * 80

        units = split_model_units(text, max_characters=32)

        self.assertEqual(
            tuple((offset, len(source)) for offset, source in units),
            ((0, 32), (32, 32), (64, 16)))
        self.assertEqual(split_model_units("", max_characters=32), ())
        self.assertEqual("".join(source for _, source in units), text)
        with self.assertRaises(ValueError):
            split_model_units("道", max_characters=31)

    def test_character_diagnostic_tokenizer_retains_exact_offsets(self):
        tokenizer = CharacterTokenizer()

        spans = tokenizer.tokenize("道 A\n德")

        self.assertEqual(
            tuple(
                (
                    span.surface,
                    span.start_offset,
                    span.end_offset,
                    span.confidence,
                )
                for span in spans),
            (
                ("道", 0, 1, 1.0),
                ("A", 2, 3, 1.0),
                ("德", 4, 5, 1.0),
            ))
        self.assertEqual(tokenizer.identity.backend, "character-debug")

    def test_ckip_decoder_groups_bi_labels_and_preserves_codepoint_offsets(
            self):
        tokenizer = self._decoder()

        spans = tokenizer._decode_chunk(
            "甲乙，𠀀丁",
            7,
            offsets=(
                (0, 0),
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (0, 0),
            ),
            label_ids=(0, 0, 1, 0, 0, 1, 0),
            scores=(0.0, 0.8, 0.6, 0.9, 0.7, 0.5, 0.0),
            attention=(1, 1, 1, 1, 1, 1, 0))

        self.assertEqual(
            tuple(
                (
                    span.surface,
                    span.start_offset,
                    span.end_offset,
                    span.confidence,
                )
                for span in spans),
            (
                ("甲乙", 7, 9, 0.7),
                ("，", 9, 10, 0.9),
                ("𠀀丁", 10, 12, 0.6),
            ))
        self.assertEqual(
            dict(tokenizer.identity.options)["decoder_version"],
            CKIP_DECODER_VERSION)

    def test_ckip_decoder_starts_new_words_for_orphan_or_discontinuous_i(
            self):
        tokenizer = self._decoder()

        spans = tokenizer._decode_chunk(
            "甲乙丙丁",
            0,
            offsets=((0, 1), (1, 2), (3, 4)),
            label_ids=(1, 1, 1),
            scores=(0.9, 0.7, 0.5),
            attention=(1, 1, 1))

        self.assertEqual(
            tuple(
                (
                    span.surface,
                    span.start_offset,
                    span.end_offset,
                )
                for span in spans),
            (
                ("甲乙", 0, 2),
                ("丁", 3, 4),
            ))

    def test_ckip_decoder_never_joins_punctuation_and_han_on_i_label(self):
        tokenizer = self._decoder()

        spans = tokenizer._decode_chunk(
            "，容夫。",
            0,
            offsets=((0, 1), (1, 2), (2, 3), (3, 4)),
            label_ids=(0, 1, 1, 1),
            scores=(0.9, 0.8, 0.7, 0.6),
            attention=(1, 1, 1, 1))

        self.assertEqual(
            tuple(span.surface for span in spans),
            ("，", "容夫", "。"))

    def test_ckip_decoder_ignores_special_tokens_but_rejects_odd_content_labels(
            self):
        tokenizer = self._decoder()
        tokenizer._model.config.id2label[2] = "LABEL_2"

        spans = tokenizer._decode_chunk(
            "道",
            0,
            offsets=((0, 0), (0, 1), (0, 0)),
            label_ids=(2, 0, 2),
            scores=(0.0, 0.9, 0.0),
            attention=(1, 1, 0))

        self.assertEqual(
            tuple(span.surface for span in spans),
            ("道",))

        with self.assertRaisesRegex(
                CorpusTokenizerUnavailableError,
                "unsupported word-segmentation label"):
            tokenizer._decode_chunk(
                "道",
                0,
                offsets=((0, 1),),
                label_ids=(2,),
                scores=(0.9,),
                attention=(1,))

    def test_ckip_long_han_refinement_is_recursive_and_offset_safe(self):
        tokenizer = self._decoder()
        calls = []

        def infer(units):
            calls.append(units)
            predictions = []
            for offset, source in units:
                if source == "甲乙丙丁戊己庚辛":
                    surfaces = ("甲乙", "丙丁戊己庚辛")
                elif source == "丙丁戊己庚辛":
                    surfaces = ("丙丁", "戊己庚辛")
                else:
                    raise AssertionError(source)
                cursor = offset
                spans = []
                for surface in surfaces:
                    end = cursor + len(surface)
                    spans.append(TokenSpan(
                        surface,
                        cursor,
                        end,
                        0.75))
                    cursor = end
                predictions.append(tuple(spans))
            return tuple(predictions)

        tokenizer._infer_units = infer
        spans = (
            TokenSpan("，", 4, 5, 0.9),
            TokenSpan("甲乙丙丁戊己庚辛", 5, 13, 0.8),
            TokenSpan("。", 13, 14, 0.9),
        )

        refined = tokenizer._refine_long_han_spans(spans)

        self.assertEqual(
            tuple(
                (
                    span.surface,
                    span.start_offset,
                    span.end_offset,
                )
                for span in refined),
            (
                ("，", 4, 5),
                ("甲乙", 5, 7),
                ("丙丁", 7, 9),
                ("戊己庚辛", 9, 13),
                ("。", 13, 14),
            ))
        self.assertEqual(
            calls,
            [
                ((5, "甲乙丙丁戊己庚辛"),),
                ((7, "丙丁戊己庚辛"),),
            ])

    def test_ckip_long_han_refinement_preserves_stable_model_span(self):
        tokenizer = self._decoder()
        original = TokenSpan("甲乙丙丁戊", 3, 8, 0.61)
        calls = []

        def infer(units):
            calls.append(units)
            return ((TokenSpan("甲乙丙丁戊", 3, 8, 0.99),),)

        tokenizer._infer_units = infer

        refined = tokenizer._refine_long_han_spans((original,))

        self.assertEqual(refined, (original,))
        self.assertEqual(calls, [((3, "甲乙丙丁戊"),)])


class ChunkTests(unittest.TestCase):
    def test_chunk_boundaries_at_499_500_501_1000_and_1001(self):
        cases = {
            499: ((1, 499, 499, "1-499"),),
            500: ((1, 500, 500, "1-500"),),
            501: (
                (1, 500, 500, "1-500"),
                (501, 501, 1, "501-501"),
            ),
            1000: (
                (1, 500, 500, "1-500"),
                (501, 1000, 500, "501-1000"),
            ),
            1001: (
                (1, 500, 500, "1-500"),
                (501, 1000, 500, "501-1000"),
                (1001, 1001, 1, "1001-1001"),
            ),
        }
        for count, expected in cases.items():
            with self.subTest(count=count):
                chunks = make_chunks(make_words(count))
                self.assertEqual(
                    tuple(
                        (
                            chunk.start_rank,
                            chunk.end_rank,
                            len(chunk.words),
                            chunk.stem,
                        )
                        for chunk in chunks),
                    expected)
                self.assertEqual(
                    tuple(
                        word.rank
                        for chunk in chunks
                        for word in chunk.words),
                    tuple(range(1, count + 1)))

    def test_empty_and_custom_size_chunks(self):
        self.assertEqual(make_chunks(()), ())
        chunks = make_chunks(make_words(5), chunk_size=2)
        self.assertEqual(
            tuple((chunk.start_rank, chunk.end_rank) for chunk in chunks),
            ((1, 2), (3, 4), (5, 5)))
        with self.assertRaises(ValueError):
            make_chunks(make_words(1), chunk_size=0)

    def test_chunk_rejects_any_noncontiguous_rank(self):
        words = list(make_words(3))
        words[1] = replace(words[1], rank=4)

        with self.assertRaises(CorpusValidationError):
            make_chunks(tuple(words))

    def test_chunk_text_and_records_include_future_context(self):
        build = make_build(chunk_size=2)
        first_chunk = make_chunks(build.unique_words, 2)[0]

        self.assertEqual(chunk_text(first_chunk), "道\n可\n")
        records = chunk_records(build, first_chunk)
        self.assertEqual(
            tuple(record["surface"] for record in records),
            ("道", "可"))
        self.assertEqual(
            tuple(record["section_title"] for record in records),
            ("第 1 節", "第 1 節"))
        self.assertEqual(
            tuple(
                record["first_occurrence_is_section_title"]
                for record in records),
            (False, False))
        self.assertEqual(
            tuple(record["sentence_text"] for record in records),
            ("道可道。", "道可道。"))
        self.assertEqual(
            tuple(record["previous_sentence_text"] for record in records),
            (None, None))
        self.assertEqual(
            tuple(record["next_sentence_text"] for record in records),
            ("名可名！", "名可名！"))
        self.assertEqual(
            tuple(record["paragraph_text"] for record in records),
            ("道可道。名可名！", "道可道。名可名！"))


class AuditTests(unittest.TestCase):
    def test_build_audit_rejects_an_empty_tokenizer_result(self):
        build = make_build()

        with self.assertRaisesRegex(
                CorpusValidationError,
                "no token occurrences"):
            audit_build(replace(
                build,
                occurrences=(),
                unique_words=()))

    def test_valid_snapshot_checks_completeness_hashes_and_latin_policy(self):
        snapshot = make_snapshot()

        self.assertTrue(audit_snapshot(
            snapshot,
            expected_section_count=2,
            expected_section_ids=("section-001", "section-002"),
            require_no_latin=True))

    def test_snapshot_rejects_expected_count_or_id_mismatch(self):
        snapshot = make_snapshot()

        with self.assertRaisesRegex(
                CorpusValidationError,
                "Expected 100 sections"):
            audit_snapshot(snapshot, expected_section_count=100)
        with self.assertRaisesRegex(
                CorpusValidationError,
                "missing, duplicated, or reordered"):
            audit_snapshot(
                snapshot,
                expected_section_ids=("section-002", "section-001"))

    def test_snapshot_rejects_raw_hash_and_duplicate_page_keys(self):
        snapshot = make_snapshot()
        bad_page = replace(snapshot.pages[0], raw_sha256="0" * 64)
        duplicate_page = replace(snapshot.pages[0], order=2)

        with self.assertRaisesRegex(
                CorpusValidationError,
                "Raw source hash failed"):
            audit_snapshot(replace(snapshot, pages=(bad_page,)))
        with self.assertRaisesRegex(
                CorpusValidationError,
                "Duplicate source page key"):
            audit_snapshot(
                replace(
                    snapshot,
                    pages=(snapshot.pages[0], duplicate_page)))

    def test_snapshot_rejects_each_clean_text_contaminant(self):
        contaminants = {
            "template opening": "道{{德",
            "template closing": "道}}德",
            "wiki-link opening": "道[[德",
            "wiki-link closing": "道]]德",
            "web address": "道https://example.invalid德",
            "category markup": "道 Category:Books 德",
            "MediaWiki directive": "道__NOTOC__德",
            "HTML/XML tag": "道<br>德",
        }
        for description, text in contaminants.items():
            with self.subTest(description=description):
                with self.assertRaisesRegex(
                        CorpusValidationError,
                        description):
                    audit_snapshot(make_snapshot((text,)))

    def test_snapshot_rejects_latin_when_requested_but_not_by_default(self):
        snapshot = make_snapshot(("道 navigation 德",))

        self.assertTrue(audit_snapshot(
            snapshot,
            require_no_latin=False))
        with self.assertRaisesRegex(
                CorpusValidationError,
                "Latin website text"):
            audit_snapshot(snapshot, require_no_latin=True)

    def test_snapshot_rejects_unknown_page_and_offset_or_text_corruption(self):
        snapshot = make_snapshot()
        section = snapshot.sections[0]
        cases = (
            (
                replace(section, source_page_key="missing"),
                "Unknown source page",
            ),
            (
                replace(
                    section,
                    start_offset=1,
                    end_offset=section.end_offset + 1),
                "Incorrect start offset",
            ),
            (
                replace(section, end_offset=section.end_offset - 1),
                "Incorrect end offset",
            ),
            (
                replace(section, text="德" + section.text[1:]),
                "Section slice mismatch",
            ),
        )
        for broken_section, message in cases:
            with self.subTest(message=message):
                broken = replace(
                    snapshot,
                    sections=(broken_section,) + snapshot.sections[1:])
                with self.assertRaisesRegex(
                        CorpusValidationError,
                        message):
                    audit_snapshot(broken)

    def test_build_audit_rejects_context_token_and_count_corruption(self):
        build = make_build()
        bad_context = replace(
            build.contexts[0],
            text="錯" + build.contexts[0].text[1:])
        bad_occurrence = replace(
            build.occurrences[0],
            surface="德")
        bad_word = replace(
            build.unique_words[0],
            occurrence_count=999)
        bad_neighbor = replace(
            build.unique_words[0],
            next_sentence_id=None)
        cases = (
            (
                replace(
                    build,
                    contexts=(bad_context,) + build.contexts[1:]),
                "Context offset mismatch",
            ),
            (
                replace(
                    build,
                    occurrences=(
                        bad_occurrence,
                    ) + build.occurrences[1:]),
                "Token offset mismatch",
            ),
            (
                replace(
                    build,
                    unique_words=(bad_word,) + build.unique_words[1:]),
                "Occurrence count mismatch",
            ),
            (
                replace(
                    build,
                    unique_words=(
                        bad_neighbor,
                    ) + build.unique_words[1:]),
                "Sentence-neighbor mismatch",
            ),
        )
        for broken, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                        CorpusValidationError,
                        message):
                    audit_build(broken)

    def test_build_audit_rejects_reordered_occurrences_and_bad_rank(self):
        build = make_build()
        reordered = (
            replace(
                build.occurrences[1],
                token_index=1,
                occurrence_id="token:000000001"),
            replace(
                build.occurrences[0],
                token_index=2,
                occurrence_id="token:000000002"),
        ) + build.occurrences[2:]
        bad_rank = replace(build.unique_words[1], rank=7)

        with self.assertRaisesRegex(
                CorpusValidationError,
                "not in source order"):
            audit_build(replace(build, occurrences=reordered))
        with self.assertRaisesRegex(
                CorpusValidationError,
                "ranks are not contiguous"):
            audit_build(
                replace(
                    build,
                    unique_words=(
                        build.unique_words[0],
                        bad_rank,
                    ) + build.unique_words[2:]))


class IdentityAndStorageTests(unittest.TestCase):
    def test_storage_rejects_traversal_or_nonportable_corpus_keys(self):
        snapshot = make_snapshot()
        bad_page = replace(
            snapshot.pages[0],
            page_key="../../escaped")
        bad_page_snapshot = replace(
            snapshot,
            pages=(bad_page,),
            sections=tuple(
                replace(
                    section,
                    source_page_key=bad_page.page_key)
                for section in snapshot.sections))
        bad_spec_snapshot = replace(
            snapshot,
            spec_key="../escaped")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for broken in (bad_page_snapshot, bad_spec_snapshot):
                with self.subTest(key=broken.spec_key):
                    with self.assertRaisesRegex(
                            ValueError,
                            "portable lowercase storage key"):
                        write_snapshot(broken, root)
            self.assertFalse(any(
                path.name == "escaped.wiki"
                for path in root.rglob("*")))

    def test_snapshot_reader_rejects_traversal_keys_in_manifest(self):
        snapshot = make_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            target = write_snapshot(snapshot, directory)
            manifest_path = target / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"))
            manifest["pages"][0]["page_key"] = "../../escaped"
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
                newline="\n")

            with self.assertRaisesRegex(
                    ValueError,
                    "portable lowercase storage key"):
                read_snapshot(target)

    def test_snapshot_reader_requires_artifacts_and_consistent_metadata(self):
        snapshot = make_snapshot()
        mutations = (
            (
                lambda manifest: manifest.pop("artifacts"),
                "artifact map",
            ),
            (
                lambda manifest: manifest.__setitem__(
                    "canonical_sha256",
                    "0" * 64),
                "canonical-text hash",
            ),
            (
                lambda manifest: manifest.__setitem__(
                    "counts",
                    {
                        "pages": 999,
                        "sections": 999,
                        "characters": 999,
                    }),
                "count metadata",
            ),
            (
                lambda manifest: manifest.__setitem__(
                    "offset_convention",
                    "UTF-16 offsets"),
                "offset convention",
            ),
        )
        for mutate, expected_message in mutations:
            with self.subTest(expected_message=expected_message):
                with tempfile.TemporaryDirectory() as directory:
                    target = write_snapshot(snapshot, directory)
                    manifest_path = target / "manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8"))
                    mutate(manifest)
                    manifest_path.write_text(
                        json.dumps(
                            manifest,
                            ensure_ascii=False,
                            sort_keys=True,
                            indent=2,
                        ) + "\n",
                        encoding="utf-8",
                        newline="\n")

                    with self.assertRaisesRegex(
                            ValueError,
                            expected_message):
                        read_snapshot(target)

    def test_storage_rejects_a_symlinked_corpus_key_directory(self):
        snapshot = make_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            outside = Path(directory) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / snapshot.spec_key).symlink_to(
                outside,
                target_is_directory=True)

            with self.assertRaisesRegex(
                    ValueError,
                    "cannot be a symlink"):
                write_snapshot(snapshot, root)
            self.assertEqual(tuple(outside.iterdir()), ())

    def test_storage_rejects_symlinked_snapshot_and_run_parents(self):
        snapshot = make_snapshot()
        build = make_build()
        cases = (
            ("snapshots", lambda root: write_snapshot(snapshot, root)),
            ("runs", lambda root: write_build(build, root)),
        )
        for parent_name, operation in cases:
            with self.subTest(parent_name=parent_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "root"
                    outside = Path(directory) / "outside"
                    corpus_base = root / snapshot.spec_key
                    corpus_base.mkdir(parents=True)
                    outside.mkdir()
                    (corpus_base / parent_name).symlink_to(
                        outside,
                        target_is_directory=True)

                    with self.assertRaisesRegex(
                            ValueError,
                            "parent directory cannot be a symlink"):
                        operation(root)
                    self.assertEqual(tuple(outside.iterdir()), ())

    def test_snapshot_and_build_ids_are_deterministic_and_content_sensitive(
            self):
        snapshot = make_snapshot()
        same_content_later = replace(
            snapshot,
            pages=(
                replace(
                    snapshot.pages[0],
                    retrieved_at="2099-01-01T00:00:00+00:00"),
            ))
        build = build_vocabulary(
            snapshot,
            FakeCharacterTokenizer(),
            BuildConfig(chunk_size=2))
        same_build = replace(build)

        self.assertEqual(snapshot_id(snapshot), snapshot_id(snapshot))
        self.assertEqual(
            snapshot_id(snapshot),
            snapshot_id(same_content_later))
        self.assertEqual(build_id(build), build_id(same_build))
        self.assertEqual(
            snapshot_id(snapshot),
            "7d48fa0086875e73c67458b1")
        self.assertEqual(
            build_id(build),
            "a4de231279a933e65c658875")
        self.assertRegex(snapshot_id(snapshot), r"^[0-9a-f]{24}$")
        self.assertRegex(build_id(build), r"^[0-9a-f]{24}$")

        changed_snapshot = make_snapshot(("道可道。",))
        self.assertNotEqual(
            snapshot_id(snapshot),
            snapshot_id(changed_snapshot))
        changed_config = replace(
            build,
            config=BuildConfig(chunk_size=3))
        changed_tokenizer = replace(
            build,
            tokenizer=replace(
                build.tokenizer,
                model_revision="fixture-2"))
        self.assertNotEqual(build_id(build), build_id(changed_config))
        self.assertNotEqual(build_id(build), build_id(changed_tokenizer))
        with patch(
                "corpus_pipeline.storage.CORPUS_PROCESSING_VERSION",
                "ordered-contextual-vocabulary-v3"):
            self.assertNotEqual(
                build_id(build),
                "eebc4b1c5df2b481831c0a20")

    def test_snapshot_serialization_round_trip_and_pointer_are_atomic(self):
        snapshot = make_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            target = write_snapshot(
                snapshot,
                root,
                expected_section_count=2,
                expected_section_ids=(
                    "section-001",
                    "section-002"),
                require_no_latin=True)
            loaded = read_snapshot(target)
            pointer_path = (
                root / snapshot.spec_key / "latest_snapshot.json")
            pointer = json.loads(
                pointer_path.read_text(encoding="utf-8"))

            self.assertEqual(loaded, snapshot)
            self.assertEqual(target.name, snapshot_id(snapshot))
            self.assertEqual(pointer["snapshot_id"], snapshot_id(snapshot))
            self.assertEqual(
                root / snapshot.spec_key / pointer["relative_path"],
                target)
            self.assertEqual(
                write_snapshot(snapshot, root),
                target)
            self.assertFalse(any(
                path.name.startswith(f".{target.name}.")
                for path in target.parent.iterdir()))

    def test_legacy_snapshot_without_artifact_map_remains_readable_for_upgrade(
            self):
        snapshot = make_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            target = write_snapshot(snapshot, directory)
            manifest_path = target / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"))
            manifest.pop("artifacts")
            manifest.pop("storage_version")
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
                newline="\n")

            self.assertEqual(read_snapshot(target), snapshot)
            with self.assertRaisesRegex(
                    ValueError,
                    "predates validated atomic reuse"):
                write_snapshot(snapshot, directory)

    def test_concurrent_identical_snapshot_publish_reuses_valid_winner(self):
        snapshot = make_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def publish_race(source, target):
                shutil.copytree(source, target)
                raise OSError(
                    errno.ENOTEMPTY,
                    "simulated concurrent directory publication")

            with patch(
                    "pathlib.Path.replace",
                    autospec=True,
                    side_effect=publish_race):
                target = write_snapshot(snapshot, root)

            self.assertEqual(read_snapshot(target), snapshot)

    def test_failed_pointer_replace_preserves_previous_snapshot_pointer(self):
        first = make_snapshot(("道。",))
        second = make_snapshot(("德。",))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_snapshot(first, root)
            pointer_path = (
                root / first.spec_key / "latest_snapshot.json")
            original_pointer = pointer_path.read_bytes()

            with patch(
                    "corpus_pipeline.storage.os.replace",
                    side_effect=OSError("simulated pointer failure")):
                with self.assertRaisesRegex(
                        OSError,
                        "simulated pointer failure"):
                    write_snapshot(second, root)

            self.assertEqual(pointer_path.read_bytes(), original_pointer)
            self.assertFalse(any(
                path.name.startswith(".latest_snapshot.json.")
                for path in pointer_path.parent.iterdir()))

    def test_read_snapshot_rejects_tampered_source_and_clean_artifacts(self):
        snapshot = make_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            target = write_snapshot(snapshot, directory)
            source_path = target / "source" / "page-001.wiki"
            source_path.write_text("tampered source", encoding="utf-8")

            with self.assertRaises(
                    (ValueError, CorpusValidationError)):
                read_snapshot(target)

        with tempfile.TemporaryDirectory() as directory:
            target = write_snapshot(snapshot, directory)
            text_path = target / "clean" / "text.txt"
            text_path.write_text("tampered clean text", encoding="utf-8")

            with self.assertRaises(
                    (ValueError, CorpusValidationError)):
                read_snapshot(target)

    def test_build_serialization_is_inspectable_and_deterministic(self):
        build = make_build(chunk_size=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            target = write_build(build, root)
            second_target = write_build(build, root)
            manifest = json.loads(
                (target / "manifest.json").read_text(encoding="utf-8"))
            pointer = json.loads(
                (
                    root
                    / build.snapshot.spec_key
                    / "latest_build.json"
                ).read_text(encoding="utf-8"))

            self.assertEqual(target, second_target)
            self.assertEqual(target.name, build_id(build))
            self.assertEqual(pointer["build_id"], build_id(build))
            self.assertEqual(manifest["kind"], "corpus_build")
            self.assertEqual(
                manifest["processing_version"],
                CORPUS_PROCESSING_VERSION)
            self.assertEqual(manifest["counts"]["unique_words"], 5)
            self.assertEqual(manifest["counts"]["chunks"], 3)
            self.assertEqual(
                tuple(
                    path.name
                    for path in sorted(
                        (target / "chunks").glob("*.txt"))),
                ("1-2.txt", "3-4.txt", "5-5.txt"))
            self.assertEqual(
                (target / "chunks" / "1-2.txt").read_text(
                    encoding="utf-8"),
                "道\n可\n")
            chunk_json = json.loads(
                (target / "chunks" / "1-2.json").read_text(
                    encoding="utf-8"))
            self.assertEqual(
                tuple(
                    record["sentence_text"]
                    for record in chunk_json["words"]),
                ("道可道。", "道可道。"))
            self.assertIn(
                "No OpenAI or Anki operation",
                (
                    target / "audit" / "report.txt"
                ).read_text(encoding="utf-8"))
            self.assertEqual(read_build(target), build)
            view = read_vocabulary_view(target)
            self.assertEqual(view.snapshot, build.snapshot)
            self.assertEqual(view.contexts, build.contexts)
            self.assertEqual(view.unique_words, build.unique_words)
            self.assertFalse(hasattr(view, "occurrences"))

    def test_existing_build_is_revalidated_before_reuse(self):
        build = make_build(chunk_size=2)
        with tempfile.TemporaryDirectory() as directory:
            target = write_build(build, directory)
            chunk = target / "chunks" / "1-2.txt"
            chunk.write_text("篡改\n", encoding="utf-8")

            with self.assertRaisesRegex(
                    ValueError,
                    "artifact hash failed"):
                write_build(build, directory)

    def test_reading_embedded_snapshot_from_build_verifies_build_artifacts(
            self):
        build = make_build(chunk_size=2)
        with tempfile.TemporaryDirectory() as directory:
            target = write_build(build, directory)
            sections_path = target / "clean" / "sections.jsonl"
            sections_path.write_text(
                sections_path.read_text(
                    encoding="utf-8").replace(
                        "第 1 節",
                        "TAMPERED TITLE"),
                encoding="utf-8",
                newline="\n")

            with self.assertRaisesRegex(
                    ValueError,
                    "artifact hash failed"):
                read_snapshot(target)

    def test_saved_build_semantics_are_audited_beyond_file_hashes(self):
        build = make_build(chunk_size=2)
        with tempfile.TemporaryDirectory() as directory:
            target = write_build(build, directory)
            words_path = target / "unique_words.jsonl"
            records = [
                json.loads(line)
                for line in words_path.read_text(
                    encoding="utf-8").splitlines()
            ]
            records[0]["paragraph_id"] = records[-1]["paragraph_id"]
            words_path.write_text(
                "".join(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                    ) + "\n"
                    for record in records
                ),
                encoding="utf-8",
                newline="\n")
            manifest_path = target / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["unique_words.jsonl"] = hashlib.sha256(
                words_path.read_bytes()).hexdigest()
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
                newline="\n")

            with self.assertRaisesRegex(
                    CorpusValidationError,
                    "Unique-word location mismatch"):
                read_build(target)

    def test_saved_chunk_content_is_checked_even_with_an_updated_hash(self):
        build = make_build(chunk_size=2)
        with tempfile.TemporaryDirectory() as directory:
            target = write_build(build, directory)
            chunk_path = target / "chunks" / "1-2.json"
            chunk = json.loads(
                chunk_path.read_text(encoding="utf-8"))
            chunk["words"][0]["sentence_text"] = "偽造內容"
            chunk_path.write_text(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
                newline="\n")
            manifest_path = target / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["chunks/1-2.json"] = hashlib.sha256(
                chunk_path.read_bytes()).hexdigest()
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
                newline="\n")

            with self.assertRaisesRegex(
                    ValueError,
                    "JSON chunk is inconsistent"):
                read_build(target)

    def test_serialized_paths_are_portable_and_source_bytes_keep_lf(self):
        snapshot = make_snapshot(
            raw_wikitext="第一行\n第二行\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = write_snapshot(snapshot, root)
            manifest = json.loads(
                (target / "manifest.json").read_text(encoding="utf-8"))
            pointer = json.loads(
                (
                    root
                    / snapshot.spec_key
                    / "latest_snapshot.json"
                ).read_text(encoding="utf-8"))

            self.assertEqual(
                (
                    target
                    / "source"
                    / "page-001.wiki"
                ).read_bytes(),
                snapshot.pages[0].raw_wikitext.encode("utf-8"))
            self.assertNotIn("\\", pointer["relative_path"])
            self.assertTrue(manifest["artifacts"])
            self.assertTrue(all(
                "\\" not in relative_path
                for relative_path in manifest["artifacts"]))


class OfflineBoundaryTests(unittest.TestCase):
    def test_processing_modules_do_not_import_paid_or_network_clients(self):
        module_names = (
            "models.py",
            "contexts.py",
            "processing.py",
            "chunks.py",
            "audit.py",
            "storage.py",
            "tokenizers.py",
            "refiners.py",
        )
        forbidden_roots = {
            "anki",
            "genanki",
            "httpx",
            "openai",
            "requests",
            "socket",
            "urllib",
        }
        found = []
        for module_name in module_names:
            path = PROJECT_ROOT / "src" / "corpus_pipeline" / module_name
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports = (node.module or "",)
                else:
                    continue
                for imported in imports:
                    root = imported.partition(".")[0]
                    if root in forbidden_roots:
                        found.append((module_name, imported))
        self.assertEqual(found, [])

    def test_build_and_serialization_work_with_forbidden_import_guard(self):
        original_import = builtins.__import__
        forbidden_roots = {
            "anki",
            "genanki",
            "httpx",
            "openai",
            "requests",
            "socket",
            "urllib",
        }

        def guarded_import(name, *args, **kwargs):
            if name.partition(".")[0] in forbidden_roots:
                raise AssertionError(
                    f"Offline corpus processing imported {name}")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch("builtins.__import__", side_effect=guarded_import):
                build = make_build(chunk_size=2)
                target = write_build(build, directory)

        self.assertTrue(target.name)


if __name__ == "__main__":
    unittest.main()
