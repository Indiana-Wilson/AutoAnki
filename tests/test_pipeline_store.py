import json
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import genanki


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
        first = replace(
            pipeline_store.default_pipeline(),
            target_deck="Retained::English",
            language_settings=(
                pipeline_store.LanguageSettings(
                    language_key="english",
                    card_type_keys=("english_vocabulary",),
                    target_deck="Retained::English"),
                pipeline_store.LanguageSettings(
                    language_key="french",
                    card_type_keys=(
                        "french_word_to_meaning",
                        "french_meaning_to_word"),
                    target_deck="Retained::French",
                    separate_target_decks=True,
                    card_type_target_decks=(
                        (
                            "french_word_to_meaning",
                            "French::Recognition"),
                        (
                            "french_meaning_to_word",
                            "French::Production"),
                    )),
            ))
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
        self.assertEqual(
            loaded[0].language_settings[1].target_deck,
            "Retained::French")

    def test_each_language_retains_its_own_deck_preferences(self):
        pipeline = replace(
            pipeline_store.default_pipeline(),
            language_settings=(
                pipeline_store.LanguageSettings(
                    language_key="french",
                    card_type_keys=(
                        "french_word_to_meaning",
                        "french_meaning_to_word"),
                    target_deck="Retained::French",
                    separate_target_decks=True,
                    card_type_target_decks=(
                        (
                            "french_word_to_meaning",
                            "French::Recognition"),
                        (
                            "french_meaning_to_word",
                            "French::Production"),
                    )),
                pipeline_store.LanguageSettings(
                    language_key="japanese",
                    card_type_keys=("japanese_vocabulary",),
                    target_deck="Retained::Japanese"),
            ))

        french = pipeline_store.get_language_settings(
            pipeline,
            "french")
        japanese = pipeline_store.get_language_settings(
            pipeline,
            "japanese")

        self.assertEqual(french.target_deck, "Retained::French")
        self.assertTrue(french.separate_target_decks)
        self.assertEqual(
            dict(french.card_type_target_decks),
            {
                "french_word_to_meaning": "French::Recognition",
                "french_meaning_to_word": "French::Production",
            })
        self.assertEqual(japanese.target_deck, "Retained::Japanese")

    def test_version_three_settings_gain_shared_deck_defaults(self):
        pipeline = pipeline_store.default_pipeline()
        legacy_data = {
            "version": 3,
            "pipelines": [{
                key: value
                for key, value in pipeline.__dict__.items()
                if key not in {
                    "separate_target_decks",
                    "card_type_target_decks",
                }
            }],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "pipelines.json"
            path.write_text(json.dumps(legacy_data), encoding="utf-8")
            loaded = pipeline_store.load_pipelines(path)

        self.assertFalse(loaded[0].separate_target_decks)
        self.assertEqual(loaded[0].card_type_target_decks, ())
        self.assertEqual(loaded[0].language_settings, ())
        self.assertEqual(
            pipeline_store.get_language_settings(
                loaded[0],
                loaded[0].language_key).target_deck,
            loaded[0].target_deck)

    def test_separate_decks_map_selected_card_types_to_model_names(self):
        pipeline = replace(
            pipeline_store.default_pipeline(),
            card_type_keys=(
                templates.ENGLISH_VOCABULARY_CARD_TYPE.key,
                templates.ENGLISH_WORD_TO_MEANING_CARD_TYPE.key,
            ),
            separate_target_decks=True,
            card_type_target_decks=(
                (
                    templates.ENGLISH_VOCABULARY_CARD_TYPE.key,
                    "English::Context"),
                (
                    templates.ENGLISH_WORD_TO_MEANING_CARD_TYPE.key,
                    "English::Recognition"),
            ),
            target_deck="")

        pipeline_store.validate_pipelines((pipeline,))

        self.assertEqual(
            pipeline_store.get_model_target_decks(pipeline),
            (
                (
                    templates.ENGLISH_VOCABULARY_CARD_TYPE.model.name,
                    "English::Context"),
                (
                    templates.ENGLISH_WORD_TO_MEANING_CARD_TYPE.model.name,
                    "English::Recognition"),
            ))

    def test_separate_decks_require_a_target_for_every_selected_card_type(self):
        pipeline = replace(
            pipeline_store.default_pipeline(),
            card_type_keys=(
                templates.ENGLISH_VOCABULARY_CARD_TYPE.key,
                templates.ENGLISH_WORD_TO_MEANING_CARD_TYPE.key,
            ),
            separate_target_decks=True,
            card_type_target_decks=((
                templates.ENGLISH_VOCABULARY_CARD_TYPE.key,
                "English::Context"),))

        with self.assertRaisesRegex(
                ValueError,
                "every selected card output"):
            pipeline_store.validate_pipelines((pipeline,))

    def test_anki_deck_cache_round_trip_is_isolated_to_supplied_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "anki_decks.json"
            saved = pipeline_store.save_anki_deck_cache(
                ("Vocabulary", "Vocabulary::English", "Vocabulary"),
                path)
            loaded = pipeline_store.load_anki_deck_cache(path)

        self.assertEqual(
            saved,
            ("Vocabulary", "Vocabulary::English"))
        self.assertEqual(loaded, saved)

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

    def test_prompt_text_is_saved_atomically_inside_prompt_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            prompt_directory = project_root / "input" / "prompts"
            prompt_directory.mkdir(parents=True)
            prompt_path = prompt_directory / "english_vocab"
            prompt_path.write_text(
                "Original prompt",
                encoding="utf-8")

            result = pipeline_store.save_prompt_text(
                prompt_path,
                "Updated prompt",
                project_root)

            self.assertEqual(result, prompt_path.resolve())
            self.assertEqual(
                prompt_path.read_text(encoding="utf-8"),
                "Updated prompt")
            self.assertEqual(
                list(prompt_directory.iterdir()),
                [prompt_path])

    def test_prompt_editor_rejects_empty_or_outside_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            prompt_directory = project_root / "input" / "prompts"
            prompt_directory.mkdir(parents=True)
            prompt_path = prompt_directory / "english_vocab"
            prompt_path.write_text(
                "Original prompt",
                encoding="utf-8")
            outside_path = project_root / "outside"
            outside_path.write_text(
                "Outside",
                encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                pipeline_store.save_prompt_text(
                    prompt_path,
                    "   ",
                    project_root)
            with self.assertRaisesRegex(ValueError, "inside input/prompts"):
                pipeline_store.save_prompt_text(
                    outside_path,
                    "Changed",
                    project_root)

            self.assertEqual(
                prompt_path.read_text(encoding="utf-8"),
                "Original prompt")
            self.assertEqual(
                outside_path.read_text(encoding="utf-8"),
                "Outside")

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

    def test_native_detailed_card_selects_rich_language_prompt(self):
        pipeline = pipeline_store.PipelineConfig(
            **{
                **pipeline_store.default_pipeline().__dict__,
                "language_key": "japanese",
                "card_type_keys": (
                    "japanese_native_vocabulary",
                    "japanese_word_to_meaning"),
            })

        self.assertEqual(
            pipeline_store.get_prompt_key(pipeline),
            "japanese_vocab")

    def test_mixed_simple_definitions_use_minimal_language_prompt(self):
        pipeline = pipeline_store.PipelineConfig(
            **{
                **pipeline_store.default_pipeline().__dict__,
                "language_key": "french",
                "card_type_keys": (
                    "french_word_to_meaning",
                    "french_word_to_native_meaning"),
            })

        self.assertEqual(
            pipeline_store.get_prompt_key(pipeline),
            "french_vocab_simple")

    def test_languages_expose_expected_compatible_card_outputs(self):
        self.assertEqual(
            {
                language.key: len(language.card_types)
                for language in pipeline_store.list_languages()
            },
            {
                "english": 3,
                "classical_chinese": 6,
                "french": 6,
                "japanese": 6,
                "latin": 6,
            })

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
                templates.CLASSICAL_CHINESE_NATIVE_VOCABULARY_MODEL_ID,
                templates
                .CLASSICAL_CHINESE_WORD_TO_NATIVE_MEANING_MODEL_ID,
                templates
                .CLASSICAL_CHINESE_NATIVE_MEANING_TO_WORD_MODEL_ID,
                templates.FRENCH_VOCABULARY_MODEL_ID,
                templates.FRENCH_WORD_TO_MEANING_MODEL_ID,
                templates.FRENCH_MEANING_TO_WORD_MODEL_ID,
                templates.FRENCH_NATIVE_VOCABULARY_MODEL_ID,
                templates.FRENCH_WORD_TO_NATIVE_MEANING_MODEL_ID,
                templates.FRENCH_NATIVE_MEANING_TO_WORD_MODEL_ID,
                templates.JAPANESE_VOCABULARY_MODEL_ID,
                templates.JAPANESE_WORD_TO_MEANING_MODEL_ID,
                templates.JAPANESE_MEANING_TO_WORD_MODEL_ID,
                templates.JAPANESE_NATIVE_VOCABULARY_MODEL_ID,
                templates.JAPANESE_WORD_TO_NATIVE_MEANING_MODEL_ID,
                templates.JAPANESE_NATIVE_MEANING_TO_WORD_MODEL_ID,
                templates.LATIN_VOCABULARY_MODEL_ID,
                templates.LATIN_WORD_TO_MEANING_MODEL_ID,
                templates.LATIN_MEANING_TO_WORD_MODEL_ID,
                templates.LATIN_NATIVE_VOCABULARY_MODEL_ID,
                templates.LATIN_WORD_TO_NATIVE_MEANING_MODEL_ID,
                templates.LATIN_NATIVE_MEANING_TO_WORD_MODEL_ID,
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
                "classical_chinese_native_vocabulary",
                "classical_chinese_word_to_native_meaning",
                "classical_chinese_native_meaning_to_word",
                "french_vocabulary",
                "french_word_to_meaning",
                "french_meaning_to_word",
                "french_native_vocabulary",
                "french_word_to_native_meaning",
                "french_native_meaning_to_word",
                "japanese_vocabulary",
                "japanese_word_to_meaning",
                "japanese_meaning_to_word",
                "japanese_native_vocabulary",
                "japanese_word_to_native_meaning",
                "japanese_native_meaning_to_word",
                "latin_vocabulary",
                "latin_word_to_meaning",
                "latin_meaning_to_word",
                "latin_native_vocabulary",
                "latin_word_to_native_meaning",
                "latin_native_meaning_to_word",
            })

    def test_every_registered_model_uses_shared_readable_card_css(self):
        models = {
            card_type.model.model_id: card_type.model
            for card_type in templates.list_card_types()
        }

        self.assertEqual(len(models), 27)
        self.assertEqual(
            {
                model.css
                for model in models.values()
            },
            {templates.CARD_CSS})
        self.assertIn(
            "font-size: 22px",
            templates.CARD_CSS)

    def test_every_registered_model_can_be_packaged_with_declared_fields(self):
        deck = templates.create_deck(
            2059400999,
            "All registered models")
        for index, card_type in enumerate(
                templates.list_card_types()):
            deck.add_note(genanki.Note(
                model=card_type.model,
                fields=[
                    f"{field_name} {index}"
                    for field_name in card_type.field_names
                ]))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "models.apkg"
            genanki.Package(deck).write_to_file(output_path)

            with zipfile.ZipFile(output_path) as package:
                self.assertIn(
                    "collection.anki2",
                    package.namelist())


if __name__ == "__main__":
    unittest.main()
