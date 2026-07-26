import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from source_generation.translation_memory import (
    SourceContextTranslationMemory,
    translation_memory_key,
)


def provenance(identifier):
    return {
        "provenance_id": identifier,
        "origin": "provider",
        "fully_validated": True,
        "manual_acceptance": False,
    }


class TranslationMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = SourceContextTranslationMemory(
            Path(self.temporary.name))

    def test_exact_validated_translation_round_trips(self):
        result = self.store.commit(
            "french",
            "Il court.\r\n",
            "He runs.",
            provenance=provenance("one"))

        hit = self.store.lookup(
            "french",
            "Il court.\n")

        self.assertEqual(result["state"], "usable")
        self.assertEqual(hit["translation"], "He runs.")
        self.assertEqual(
            hit["cache_key"],
            translation_memory_key("french", "Il court.\n"))

    def test_language_whitespace_case_and_nfkc_do_not_fuzzy_match(self):
        self.store.commit(
            "french",
            "École",
            "School",
            provenance=provenance("one"))

        self.assertIsNone(self.store.lookup("english", "École"))
        self.assertIsNone(self.store.lookup("french", "école"))
        self.assertIsNone(self.store.lookup("french", " École"))
        self.assertIsNone(self.store.lookup("french", "Ｅ́cole"))

    def test_distinct_validated_variant_marks_conflict_and_disables_hit(self):
        self.store.commit(
            "japanese",
            "走る。",
            "Run.",
            provenance=provenance("one"))
        result = self.store.commit(
            "japanese",
            "走る。",
            "He runs.",
            provenance=provenance("two"))

        self.assertEqual(result["state"], "conflicted")
        self.assertEqual(result["variant_count"], 2)
        self.assertIsNone(self.store.lookup("japanese", "走る。"))

    def test_manual_or_memory_origin_cannot_be_admitted(self):
        for invalid in (
                {
                    "origin": "provider",
                    "fully_validated": True,
                    "manual_acceptance": True,
                },
                {
                    "origin": "memory",
                    "fully_validated": True,
                    "manual_acceptance": False,
                }):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.store.commit(
                        "english",
                        "A sentence.",
                        "A sentence.",
                        provenance=invalid)

    def test_corrupt_entry_is_a_cache_miss(self):
        self.store.commit(
            "english",
            "A sentence.",
            "A sentence.",
            provenance=provenance("one"))
        cache_key = translation_memory_key(
            "english",
            "A sentence.")
        entry_path, _lock_path = self.store._paths(cache_key)
        entry_path.write_text(
            json.dumps({"schema_version": 999}),
            encoding="utf-8")

        self.assertIsNone(
            self.store.lookup("english", "A sentence."))


if __name__ == "__main__":
    unittest.main()
