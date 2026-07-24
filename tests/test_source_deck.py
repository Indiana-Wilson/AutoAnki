import sqlite3
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
import source_deck


class SourceDeckTests(unittest.TestCase):
    def test_deck_identity_is_stable_and_source_specific(self):
        self.assertEqual(
            source_deck.source_deck_id("source-one"),
            source_deck.source_deck_id("source-one"))
        self.assertNotEqual(
            source_deck.source_deck_id("source-one"),
            source_deck.source_deck_id("source-two"))
        self.assertEqual(
            source_deck.source_deck_name("  Journey   to the West "),
            "Vocabulary from Journey to the West")

    def test_combined_response_is_packaged_into_source_named_deck(self):
        pipeline = pipeline_store.default_pipeline()
        response = {
            "cards": [
                {
                    "Word": "flourish",
                    "Sentences": (
                        "Plants <strong>flourish</strong> here.|"
                        "Arts <strong>flourish</strong> in peace.|"
                        "Trade can <strong>flourish</strong>.|"
                        "May learning <strong>flourish</strong>."),
                    "Dictionary Meaning (English)": (
                        "To grow or develop successfully."),
                    "Pronunciation (English)": "IPA: /ˈflʌrɪʃ/",
                },
                {
                    "Word": "fulfil",
                    "Sentences": (
                        "They <strong>fulfil</strong> the promise.|"
                        "The result <strong>fulfils</strong> the need.|"
                        "We <strong>fulfil</strong> our duties.|"
                        "This will <strong>fulfil</strong> the goal."),
                    "Dictionary Meaning (English)": (
                        "To carry out or bring to completion."),
                    "Pronunciation (English)": "IPA: /fʊlˈfɪl/",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.apkg"
            package_path, count = source_deck.create_source_package(
                response,
                source_title="A Work",
                source_key="work-key",
                pipeline=pipeline,
                output_path=output)

            self.assertEqual(package_path, output)
            self.assertEqual(count, 2)
            self.assertTrue(output.is_file())
            with zipfile.ZipFile(output) as package:
                package.extract("collection.anki2", path=directory)
            database = sqlite3.connect(
                Path(directory) / "collection.anki2")
            try:
                due_positions = tuple(
                    row[0]
                    for row in database.execute(
                        "SELECT due FROM cards ORDER BY id"))
            finally:
                database.close()
            self.assertEqual(due_positions, (1, 2))

    def test_import_uses_standalone_workflow(self):
        importer = MagicMock(return_value=True)
        client = MagicMock()

        result = source_deck.import_source_package(
            "/tmp/source.apkg",
            client=client,
            importer=importer)

        self.assertTrue(result)
        importer.assert_called_once_with(
            "/tmp/source.apkg",
            client=client)


if __name__ == "__main__":
    unittest.main()
