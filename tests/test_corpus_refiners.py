import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from corpus_pipeline.audit import audit_build
from corpus_pipeline.contexts import assemble_sections
from corpus_pipeline.models import (
    BuildConfig,
    CorpusSnapshot,
    SourcePage,
    TextSection,
    TokenSpan,
    TokenizerIdentity,
)
from corpus_pipeline.processing import build_vocabulary
from corpus_pipeline.refiners import (
    JIEBA_REFINER_VERSION,
    JiebaLongSpanRefiner,
    refine_build_long_spans,
)
from corpus_pipeline.resources import (
    ensure_jieba_traditional_dictionary,
)


class FixedTokenizer:
    @property
    def identity(self):
        return TokenizerIdentity(
            backend="fixture",
            backend_version="1",
            model="fixture-long-spans",
            model_revision="1",
        )

    def tokenize(self, _text):
        return (
            TokenSpan("今有花果山", 0, 5, 0.75),
            TokenSpan("強中更有強中手", 6, 13, 0.8),
        )


class FakeJieba:
    def cut(self, surface, *, HMM):
        if HMM:
            raise AssertionError("Unknown-word guessing must stay disabled.")
        if surface == "今有花果山":
            return iter(("今", "有", "花果山"))
        if surface == "強中更有強中手":
            return iter((surface,))
        raise AssertionError(surface)


def make_build():
    text = "今有花果山。強中更有強中手。"
    raw = f"== fixture ==\n{text}"
    page = SourcePage(
        page_key="page-001",
        order=1,
        title="Fixture",
        url="https://example.invalid/fixture",
        revision_id=1,
        revision_timestamp="2026-07-24T00:00:00Z",
        revision_sha1="fixture",
        retrieved_at="2026-07-24T00:00:00Z",
        raw_wikitext=raw,
        raw_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )
    sections, canonical_text = assemble_sections((
        TextSection(
            section_id="fixture:chapter:001",
            order=1,
            title="Fixture",
            source_page_key=page.page_key,
            text=text,
        ),
    ))
    snapshot = CorpusSnapshot(
        spec_key="fixture",
        edition="Fixture",
        source_language_key="classical_chinese_ming",
        pages=(page,),
        sections=sections,
        canonical_text=canonical_text,
        cleaner_version="fixture",
    )
    return build_vocabulary(
        snapshot,
        FixedTokenizer(),
        BuildConfig(),
    )


class DictionaryResourceTests(unittest.TestCase):
    def test_resource_is_hashed_cached_and_manifested_without_refetch(self):
        payload = b"fixture Traditional dictionary\n"
        expected_hash = hashlib.sha256(payload).hexdigest()
        calls = []

        def download(url):
            calls.append(url)
            return payload

        with tempfile.TemporaryDirectory() as directory:
            first = ensure_jieba_traditional_dictionary(
                directory,
                download_bytes=download,
                expected_sha256=expected_hash,
                source_url="https://example.invalid/dictionary",
                source_revision="fixture-revision",
            )
            second = ensure_jieba_traditional_dictionary(
                directory,
                download_bytes=lambda _url: self.fail(
                    "A valid cached dictionary must not be fetched again."),
                expected_sha256=expected_hash,
                source_url="https://example.invalid/dictionary",
                source_revision="fixture-revision",
            )
            manifest = json.loads(
                (first.parent / "manifest.json").read_text(
                    encoding="utf-8"))

            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), payload)
            self.assertEqual(
                calls,
                ["https://example.invalid/dictionary"],
            )
            self.assertEqual(manifest["sha256"], expected_hash)
            self.assertEqual(
                manifest["source_revision"],
                "fixture-revision",
            )

    def test_resource_rejects_wrong_or_tampered_content(self):
        payload = b"fixture dictionary"
        expected_hash = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "failed its hash"):
                ensure_jieba_traditional_dictionary(
                    directory,
                    download_bytes=lambda _url: b"wrong",
                    expected_sha256=expected_hash,
                )

            path = ensure_jieba_traditional_dictionary(
                directory,
                download_bytes=lambda _url: payload,
                expected_sha256=expected_hash,
            )
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "failed its hash"):
                ensure_jieba_traditional_dictionary(
                    directory,
                    download_bytes=lambda _url: self.fail(
                        "Tampering must not be silently overwritten."),
                    expected_sha256=expected_hash,
                )

    def test_resource_storage_rejects_a_symlink_escape(self):
        payload = b"fixture dictionary"
        expected_hash = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            (root / "_resources").symlink_to(
                outside,
                target_is_directory=True,
            )

            with self.assertRaisesRegex(ValueError, "cannot be a symlink"):
                ensure_jieba_traditional_dictionary(
                    root,
                    download_bytes=lambda _url: payload,
                    expected_sha256=expected_hash,
                )

    def test_resource_rejects_a_traversal_hash_before_creating_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpora"
            outside = Path(directory) / "escape"
            traversal = "../../../../escape/" + ("a" * 64)

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                ensure_jieba_traditional_dictionary(
                    root,
                    download_bytes=lambda _url: self.fail(
                        "An invalid hash must fail before downloading."),
                    expected_sha256=traversal,
                )

            self.assertFalse(root.exists())
            self.assertFalse(outside.exists())


class JiebaRefinerTests(unittest.TestCase):
    def test_long_spans_are_split_reindexed_and_ranked_by_first_use(self):
        dictionary = b"fixture"
        dictionary_hash = hashlib.sha256(dictionary).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dict.txt.big"
            path.write_bytes(dictionary)
            refiner = JiebaLongSpanRefiner(
                path,
                segmenter=FakeJieba(),
                package_version="fixture",
                dictionary_sha256=dictionary_hash,
                dictionary_revision="fixture-revision",
            )

            refined = refine_build_long_spans(make_build(), refiner)

        self.assertEqual(
            tuple(item.surface for item in refined.occurrences),
            (
                "今", "有", "花果山",
                "強", "中", "更", "有", "強", "中", "手",
            ),
        )
        self.assertEqual(
            tuple(item.surface for item in refined.unique_words),
            ("今", "有", "花果山", "強", "中", "更", "手"),
        )
        self.assertEqual(
            tuple(item.occurrence_count for item in refined.unique_words),
            (1, 2, 1, 2, 2, 1, 1),
        )
        self.assertEqual(
            tuple(item.token_index for item in refined.occurrences),
            tuple(range(1, 11)),
        )
        self.assertEqual(
            tuple(
                (item.start_offset, item.end_offset)
                for item in refined.occurrences),
            (
                (0, 1), (1, 2), (2, 5),
                (6, 7), (7, 8), (8, 9), (9, 10),
                (10, 11), (11, 12), (12, 13),
            ),
        )
        options = dict(refined.tokenizer.options)
        self.assertEqual(
            options["dictionary_refiner_version"],
            JIEBA_REFINER_VERSION,
        )
        self.assertEqual(options["dictionary_refiner_hmm"], "false")
        self.assertTrue(audit_build(refined))
        self.assertIs(
            refine_build_long_spans(refined, refiner),
            refined,
        )

    def test_refiner_rejects_a_nonpartitioning_segmenter(self):
        class BadSegmenter:
            def cut(self, _surface, *, HMM):
                return iter(("遺失",))

        dictionary = b"fixture"
        dictionary_hash = hashlib.sha256(dictionary).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dict.txt.big"
            path.write_bytes(dictionary)
            refiner = JiebaLongSpanRefiner(
                path,
                segmenter=BadSegmenter(),
                dictionary_sha256=dictionary_hash,
            )
            with self.assertRaisesRegex(ValueError, "preserve"):
                refiner.split("今有花果山")


if __name__ == "__main__":
    unittest.main()
