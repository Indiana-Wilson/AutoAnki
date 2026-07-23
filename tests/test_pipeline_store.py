import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
import templates


class PipelineStoreTests(unittest.TestCase):
    def test_missing_settings_return_stable_default_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "missing.json"
            pipelines = pipeline_store.load_pipelines(path)

        self.assertEqual(
            pipelines,
            (pipeline_store.default_pipeline(),))
        self.assertEqual(
            pipelines[0].generated_deck_id,
            templates.DECK_ID)

    def test_pipeline_settings_round_trip(self):
        first = pipeline_store.default_pipeline()
        second = pipeline_store.create_pipeline((first,))

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "pipelines.json"
            pipeline_store.save_pipelines(
                (first, second),
                path)
            loaded = pipeline_store.load_pipelines(path)
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, (first, second))
        self.assertEqual(
            raw["version"],
            pipeline_store.PIPELINE_CONFIG_VERSION)

    def test_new_pipeline_gets_persistent_unique_deck_identity(self):
        first = pipeline_store.default_pipeline()
        second = pipeline_store.create_pipeline((first,))

        self.assertNotEqual(
            first.pipeline_id,
            second.pipeline_id)
        self.assertNotEqual(
            first.generated_deck_id,
            second.generated_deck_id)
        self.assertGreaterEqual(
            second.generated_deck_id,
            1 << 30)
        self.assertLess(
            second.generated_deck_id,
            1 << 31)

    def test_duplicate_deck_ids_are_rejected(self):
        first = pipeline_store.default_pipeline()
        duplicate = pipeline_store.PipelineConfig(
            pipeline_id="another-pipeline",
            language_key=first.language_key,
            card_type_keys=first.card_type_keys,
            target_deck=first.target_deck,
            generated_deck_id=first.generated_deck_id,
            generated_deck_name="Another generated deck")

        with self.assertRaisesRegex(ValueError, "deck IDs"):
            pipeline_store.validate_pipelines(
                (first, duplicate))

    def test_prompt_discovery_uses_only_the_prompts_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            input_directory = project_root / "input"
            prompt_directory = input_directory / "prompts"
            prompt_directory.mkdir(parents=True)
            (prompt_directory / "english_vocab").write_text(
                "English vocabulary",
                encoding="utf-8")
            (prompt_directory / "historical_events.txt").write_text(
                "history",
                encoding="utf-8")

            prompts = pipeline_store.discover_prompts(project_root)

        self.assertEqual(
            [prompt.key for prompt in prompts],
            ["english_vocab", "historical_events.txt"])

    def test_version_one_pipeline_keys_are_migrated(self):
        legacy_data = {
            "version": 1,
            "pipelines": [{
                "pipeline_id": "default-english-vocabulary",
                "prompt_key": "default",
                "card_type_key": "english_vocabulary",
                "target_deck": "English",
                "generated_deck_id": templates.DECK_ID,
                "generated_deck_name": templates.DECK_NAME,
            }],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "pipelines.json"
            path.write_text(json.dumps(legacy_data), encoding="utf-8")
            loaded = pipeline_store.load_pipelines(path)

        self.assertEqual(
            loaded[0].language_key,
            "english")
        self.assertEqual(
            loaded[0].card_type_keys,
            ("english_vocabulary",))

    def test_card_type_from_another_language_is_rejected(self):
        pipeline = pipeline_store.default_pipeline()
        mismatched = pipeline_store.PipelineConfig(
            **{
                **pipeline.__dict__,
                "card_type_keys": (
                    "classical_chinese_word_to_meaning",),
            })

        with self.assertRaisesRegex(ValueError, "not available for English"):
            pipeline_store.validate_pipelines((mismatched,))

    def test_prompt_selection_uses_minimum_required_schema(self):
        simple = pipeline_store.PipelineConfig(
            **{
                **pipeline_store.default_pipeline().__dict__,
                "card_type_keys": (
                    "english_word_to_meaning",
                    "english_meaning_to_word"),
            })
        detailed = pipeline_store.PipelineConfig(
            **{
                **simple.__dict__,
                "card_type_keys": (
                    *simple.card_type_keys,
                    "english_vocabulary"),
            })

        self.assertEqual(
            pipeline_store.get_prompt_key(simple),
            "english_vocab_simple")
        self.assertEqual(
            pipeline_store.get_prompt_key(detailed),
            "english_vocab")

    def test_languages_expose_three_compatible_card_outputs_each(self):
        self.assertEqual(
            {
                language.key: len(language.card_types)
                for language in pipeline_store.list_languages()
            },
            {"english": 3, "classical_chinese": 3})

    def test_registered_card_type_model_ids_are_unique_and_stable(self):
        model_ids = [
            card_type.model.model_id
            for card_type in templates.list_card_types()
        ]

        self.assertEqual(len(model_ids), len(set(model_ids)))
        self.assertEqual(
            set(model_ids),
            {
                templates.MODEL_ID,
                templates.CLASSICAL_CHINESE_MODEL_ID,
                templates.ENGLISH_WORD_TO_MEANING_MODEL_ID,
                templates.ENGLISH_MEANING_TO_WORD_MODEL_ID,
                templates.CLASSICAL_CHINESE_WORD_TO_MEANING_MODEL_ID,
                templates.CLASSICAL_CHINESE_MEANING_TO_WORD_MODEL_ID,
            })
        self.assertEqual(
            {
                card_type.key
                for card_type in templates.list_card_types()
            },
            {
                "english_vocabulary",
                "english_word_to_meaning",
                "english_meaning_to_word",
                "classical_chinese_vocabulary",
                "classical_chinese_word_to_meaning",
                "classical_chinese_meaning_to_word",
            })


if __name__ == "__main__":
    unittest.main()
