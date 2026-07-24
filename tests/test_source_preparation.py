import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from corpus_pipeline.models import TokenSpan, TokenizerIdentity
from corpus_pipeline.storage import read_build
import source_preparation


class FakeCharacterTokenizer:
    @property
    def identity(self):
        return TokenizerIdentity(
            backend="test-characters",
            backend_version="1",
            model="fixture",
            model_revision="1")

    def tokenize(self, text):
        return tuple(
            TokenSpan(character, index, index + 1, 1.0)
            for index, character in enumerate(text)
            if not character.isspace())


class FlakyCountingTokenizer:
    def __init__(self, *, fail_text=None):
        self.fail_text = fail_text
        self.calls = []

    @property
    def identity(self):
        return TokenizerIdentity(
            backend="test-flaky-characters",
            backend_version="1",
            model="fixture",
            model_revision="1")

    def tokenize(self, text):
        self.calls.append(text)
        if text == self.fail_text:
            raise RuntimeError("injected late tokenizer failure")
        return tuple(
            TokenSpan(character, index, index + 1, 0.75)
            for index, character in enumerate(text)
            if not character.isspace())


class SourcePreparationTests(unittest.TestCase):
    def test_utf8_source_is_processed_registered_and_reusable(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "道德經 excerpt.txt"
            source.write_text(
                "道可道。名可名。\f無名天地之始。",
                encoding="utf-8")
            corpus_root = root / "corpora"

            prepared = source_preparation.prepare_source_file(
                source,
                title="Study Text",
                language_key="classical_chinese_warring_states",
                corpus_root=corpus_root,
                tokenizer=FakeCharacterTokenizer(),
                refiner=None,
                progress_callback=events.append)

            loaded = read_build(prepared.build_path)
            listed = source_preparation.list_prepared_sources(corpus_root)
            record = json.loads(
                (
                    corpus_root
                    / "_custom_sources"
                    / f"{prepared.source_key}.json"
                ).read_text(encoding="utf-8"))

            self.assertEqual(prepared.section_count, 2)
            self.assertEqual(
                tuple(word.surface for word in loaded.unique_words),
                ("道", "可", "名", "無", "天", "地", "之", "始"))
            self.assertEqual(listed, (prepared,))
            self.assertFalse(Path(record["build_path"]).is_absolute())
            self.assertEqual(
                (
                    corpus_root
                    / prepared.source_key
                    / "original"
                    / source.name
                ).read_bytes(),
                source.read_bytes())
            self.assertEqual(events[-1].phase, "write")
            self.assertIn("8 unique", events[-1].message)

    def test_document_identity_depends_on_title_and_source_hash(self):
        key_one = source_preparation._storage_key(
            "My Source",
            "a" * 64)
        key_two = source_preparation._storage_key(
            "My Source",
            "b" * 64)
        key_three = source_preparation._storage_key(
            "Other",
            "a" * 64)

        self.assertNotEqual(key_one, key_two)
        self.assertNotEqual(key_one, key_three)
        self.assertRegex(key_one, r"^custom-[a-z0-9-]+$")

    def test_plain_text_requires_utf8_and_supported_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latin_one = root / "legacy.txt"
            latin_one.write_bytes(b"\xff")
            unsupported = root / "source.docx"
            unsupported.write_bytes(b"not a document")

            with self.assertRaisesRegex(
                    ValueError,
                    "UTF-8"):
                source_preparation.extract_source_sections(latin_one)
            with self.assertRaisesRegex(
                    ValueError,
                    "PDF"):
                source_preparation.extract_source_sections(unsupported)

    def test_image_only_pdf_stops_instead_of_silently_producing_nothing(self):
        from pypdf import PdfWriter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with path.open("wb") as output:
                writer.write(output)

            with self.assertRaisesRegex(
                    ValueError,
                    "OCR"):
                source_preparation.extract_source_sections(path)

    def test_prepared_source_registry_rejects_escaping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "_custom_sources"
            registry.mkdir()
            (registry / "custom-bad.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "source_key": "custom-bad",
                    "snapshot_path": "../outside",
                    "build_path": "../outside",
                }),
                encoding="utf-8")

            with self.assertRaisesRegex(
                    ValueError,
                    "escapes"):
                source_preparation.list_prepared_sources(root)

    def test_retry_reuses_completed_section_checkpoints(self):
        fixed_time = "2026-07-24T00:00:00+00:00"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "three-sections.txt"
            source.write_text(
                "道可道。\f名可名。\f天地始。",
                encoding="utf-8")
            cached_root = root / "cached"
            fresh_root = root / "fresh"

            first = FlakyCountingTokenizer(fail_text="天地始。")
            with (
                    mock.patch.object(
                        source_preparation,
                        "_utc_now",
                        return_value=fixed_time),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "injected late")):
                source_preparation.prepare_source_file(
                    source,
                    title="Checkpoint Fixture",
                    language_key=(
                        "classical_chinese_warring_states"),
                    corpus_root=cached_root,
                    tokenizer=first,
                    refiner=None)
            self.assertEqual(
                first.calls,
                ["道可道。", "名可名。", "天地始。"])

            retry = FlakyCountingTokenizer()
            with mock.patch.object(
                    source_preparation,
                    "_utc_now",
                    return_value=fixed_time):
                cached = source_preparation.prepare_source_file(
                    source,
                    title="Checkpoint Fixture",
                    language_key=(
                        "classical_chinese_warring_states"),
                    corpus_root=cached_root,
                    tokenizer=retry,
                    refiner=None)

            # Only the section whose first attempt raised is sent through the
            # tokenizer again. Earlier atomic checkpoints are validated and
            # reused in source order.
            self.assertEqual(retry.calls, ["天地始。"])

            uncached_tokenizer = FlakyCountingTokenizer()
            with mock.patch.object(
                    source_preparation,
                    "_utc_now",
                    return_value=fixed_time):
                uncached = source_preparation.prepare_source_file(
                    source,
                    title="Checkpoint Fixture",
                    language_key=(
                        "classical_chinese_warring_states"),
                    corpus_root=fresh_root,
                    tokenizer=uncached_tokenizer,
                    refiner=None)
            self.assertEqual(
                uncached_tokenizer.calls,
                ["道可道。", "名可名。", "天地始。"])
            self.assertEqual(
                read_build(cached.build_path),
                read_build(uncached.build_path))

    def test_corrupt_checkpoint_is_recomputed_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "two-sections.txt"
            source.write_text("道可道。\f名可名。", encoding="utf-8")
            corpus_root = root / "corpora"

            source_preparation.prepare_source_file(
                source,
                title="Corruption Fixture",
                language_key="classical_chinese_warring_states",
                corpus_root=corpus_root,
                tokenizer=FlakyCountingTokenizer(),
                refiner=None)
            checkpoint_files = sorted(
                corpus_root.glob(
                    "custom-*/checkpoints/tokenization/*/*/"
                    "section-*.json"))
            self.assertEqual(len(checkpoint_files), 2)
            corrupt = json.loads(
                checkpoint_files[0].read_text(encoding="utf-8"))
            corrupt["tokens"][0]["surface"] = "壞"
            checkpoint_files[0].write_text(
                json.dumps(corrupt, ensure_ascii=False),
                encoding="utf-8")

            tokenizer = FlakyCountingTokenizer()
            prepared = source_preparation.prepare_source_file(
                source,
                title="Corruption Fixture",
                language_key="classical_chinese_warring_states",
                corpus_root=corpus_root,
                tokenizer=tokenizer,
                refiner=None)

            self.assertEqual(tokenizer.calls, ["道可道。"])
            self.assertEqual(
                tuple(
                    word.surface
                    for word in read_build(
                        prepared.build_path).unique_words),
                ("道", "可", "名"))

    def test_cached_confidence_and_identity_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "two-sections.txt"
            source.write_text("道可道。\f名可名。", encoding="utf-8")
            corpus_root = root / "corpora"
            source_preparation.prepare_source_file(
                source,
                title="Identity Fixture",
                language_key="classical_chinese_warring_states",
                corpus_root=corpus_root,
                tokenizer=FlakyCountingTokenizer(),
                refiner=None)
            checkpoints = sorted(corpus_root.glob(
                "custom-*/checkpoints/tokenization/*/*/"
                "section-*.json"))

            invalid_confidence = json.loads(
                checkpoints[0].read_text(encoding="utf-8"))
            invalid_confidence["tokens"][0]["confidence"] = 1.5
            checkpoints[0].write_text(
                json.dumps(invalid_confidence, ensure_ascii=False),
                encoding="utf-8")
            wrong_identity = json.loads(
                checkpoints[1].read_text(encoding="utf-8"))
            wrong_identity["metadata"]["tokenizer"][
                "model_revision"
            ] = "forged"
            checkpoints[1].write_text(
                json.dumps(wrong_identity, ensure_ascii=False),
                encoding="utf-8")

            tokenizer = FlakyCountingTokenizer()
            source_preparation.prepare_source_file(
                source,
                title="Identity Fixture",
                language_key="classical_chinese_warring_states",
                corpus_root=corpus_root,
                tokenizer=tokenizer,
                refiner=None)

            self.assertEqual(tokenizer.calls, ["道可道。", "名可名。"])


if __name__ == "__main__":
    unittest.main()
