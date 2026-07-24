import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import learned_filter_store


class LearnedFilterStoreTests(unittest.TestCase):
    def test_each_language_retains_independent_sources_and_enabled_state(self):
        english = learned_filter_store.LanguageLearnedFilter(
            language_key="english",
            enabled=True,
            sources=(
                learned_filter_store.LearnedWordSource(
                    "English deck",
                    "English Vocabulary",
                    "Word"),
            ))
        french = learned_filter_store.LanguageLearnedFilter(
            language_key="french",
            enabled=False,
            sources=(
                learned_filter_store.LearnedWordSource(
                    "French deck",
                    "French Vocabulary",
                    "French"),
            ))
        settings = learned_filter_store.replace_language_filter((), english)
        settings = learned_filter_store.replace_language_filter(
            settings,
            french)

        self.assertTrue(
            learned_filter_store.get_language_filter(
                settings,
                "english").enabled)
        self.assertEqual(
            learned_filter_store.get_language_filter(
                settings,
                "french").sources[0].deck_name,
            "French deck")
        self.assertEqual(
            learned_filter_store.settings_language_key(
                "classical_chinese_ming"),
            "classical_chinese")

    def test_new_row_copies_the_previous_combination(self):
        previous = learned_filter_store.LearnedWordSource(
            "Languages",
            "Vocabulary",
            "Expression")

        copied = learned_filter_store.source_for_new_row((previous,))

        self.assertEqual(copied, previous)
        self.assertIsNot(copied, previous)
        self.assertEqual(
            learned_filter_store.source_for_new_row(()),
            learned_filter_store.LearnedWordSource())

    def test_multiple_rows_round_trip_outside_the_repository(self):
        settings = (
            learned_filter_store.LanguageLearnedFilter(
                language_key="japanese",
                enabled=True,
                sources=(
                    learned_filter_store.LearnedWordSource(
                        "Recognition",
                        "Japanese",
                        "Expression"),
                    learned_filter_store.LearnedWordSource(
                        "Production",
                        "Japanese",
                        "Expression"),
                )),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learned_word_filters.json"
            learned_filter_store.save_language_filters(
                settings,
                path)

            loaded = learned_filter_store.load_language_filters(path)

        self.assertEqual(loaded, settings)

    def test_incomplete_rows_can_be_saved_but_not_used_as_exclusions(self):
        source = learned_filter_store.LearnedWordSource(
            deck_name="Known",
            note_type="Vocabulary")
        self.assertFalse(source.complete)
        with self.assertRaisesRegex(ValueError, "deck, note type, and field"):
            source.to_exclusion_mapping()

    def test_rejects_duplicate_language_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learned_word_filters.json"
            path.write_text(
                json.dumps({
                    "version": 1,
                    "languages": [
                        {"language_key": "english"},
                        {"language_key": "english"},
                    ],
                }),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only one"):
                learned_filter_store.load_language_filters(path)


if __name__ == "__main__":
    unittest.main()
