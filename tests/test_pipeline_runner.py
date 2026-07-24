import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anki_integration
import pipeline_runner
import pipeline_store
import templates


def configure_pipeline(
        pipeline,
        language_key,
        *,
        enabled=("context",),
        fields=None,
        separate=False):
    settings = pipeline_store.get_language_settings(
        pipeline,
        language_key)
    fields = tuple(fields or settings.shared_fields)
    settings = replace(
        settings,
        cards=tuple(
            replace(
                card,
                enabled=card.direction_key in enabled,
                fields=fields,
                target_deck=f"Target::{direction_index}")
            for direction_index, card in enumerate(settings.cards)),
        target_deck=f"Target::{language_key}",
        separate_target_decks=separate,
        share_field_settings=True,
        shared_fields=fields)
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key=language_key)


class PipelineRunnerTests(unittest.TestCase):
    def setUp(self):
        self.generator = MagicMock(
            side_effect=lambda _words, **kwargs: kwargs["output_path"])
        self.importer = MagicMock(
            side_effect=lambda _path, **kwargs: (
                anki_integration.AnkiImportResult(
                    cards_moved=2,
                    target_deck=kwargs["target_deck"])))

    def test_pipeline_composes_prompt_generates_and_imports(self):
        pipeline = configure_pipeline(
            pipeline_store.default_pipeline(),
            "french",
            fields=(
                pipeline_store.FieldSetting(
                    "translation",
                    "english"),
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "french"),
            ))
        progress = []

        with tempfile.TemporaryDirectory() as directory:
            summary = pipeline_runner.run_pipelines(
                "épanouir",
                (pipeline,),
                project_root=PROJECT_ROOT,
                output_root=directory,
                progress_callback=progress.append,
                openai_client="openai-client",
                anki_client="anki-client",
                generator=self.generator,
                importer=self.importer)

        self.assertEqual(len(summary.successes), 1)
        self.assertEqual(summary.cards_moved, 2)
        self.assertEqual(
            [item.stage for item in progress],
            ["generating", "importing"])
        generation = self.generator.call_args.kwargs
        self.assertIs(generation["pipeline"], pipeline)
        self.assertEqual(
            generation["guid_seed"],
            pipeline.pipeline_id)
        self.assertIn(
            'For "Translation (English)"',
            generation["prompt_text"])
        self.assertIn(
            'For "Dictionary Meaning (French)"',
            generation["prompt_text"])
        self.assertIs(
            generation["client"],
            "openai-client")
        self.importer.assert_called_once_with(
            generation["output_path"],
            target_deck="Target::french",
            source_deck=pipeline.generated_deck_name,
            client="anki-client")

    def test_two_configurations_each_make_one_generation_call(self):
        first = configure_pipeline(
            pipeline_store.default_pipeline(),
            "english",
            enabled=("word_to_meaning",),
            fields=(
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "english"),
            ))
        second = configure_pipeline(
            pipeline_store.create_pipeline((first,)),
            "classical_chinese",
            enabled=("meaning_to_word",),
            fields=(
                pipeline_store.FieldSetting(
                    "translation",
                    "english"),
            ))

        with tempfile.TemporaryDirectory() as directory:
            summary = pipeline_runner.run_pipelines(
                "word",
                (first, second),
                project_root=PROJECT_ROOT,
                output_root=directory,
                generator=self.generator,
                importer=self.importer)

        self.assertEqual(len(summary.successes), 2)
        self.assertEqual(self.generator.call_count, 2)
        self.assertEqual(self.importer.call_count, 2)
        self.assertNotEqual(
            self.generator.call_args_list[0].kwargs["deck_id"],
            self.generator.call_args_list[1].kwargs["deck_id"])

    def test_multiple_card_directions_share_one_paid_request(self):
        pipeline = configure_pipeline(
            pipeline_store.default_pipeline(),
            "japanese",
            enabled=(
                "context",
                "word_to_meaning",
                "meaning_to_word"),
            fields=(
                pipeline_store.FieldSetting(
                    "translation",
                    "english"),
                pipeline_store.FieldSetting(
                    "nuance",
                    "japanese"),
            ))

        with tempfile.TemporaryDirectory() as directory:
            pipeline_runner.run_pipelines(
                "開く",
                (pipeline,),
                project_root=PROJECT_ROOT,
                output_root=directory,
                generator=self.generator,
                importer=self.importer)

        self.generator.assert_called_once()
        prompt = self.generator.call_args.kwargs["prompt_text"]
        self.assertEqual(
            prompt.count('For "Translation (English)"'),
            1)
        self.assertEqual(
            prompt.count('For "Nuance (Japanese)"'),
            1)

    def test_per_card_fields_are_unioned_into_one_minimal_request(self):
        pipeline = pipeline_store.default_pipeline()
        settings = pipeline_store.get_language_settings(
            pipeline,
            "latin")
        cards = tuple(
            replace(
                card,
                enabled=True,
                fields=(
                    (
                        pipeline_store.FieldSetting(
                            "translation",
                            "english"),
                    )
                    if card.direction_key != "meaning_to_word"
                    else (
                        pipeline_store.FieldSetting(
                            "dictionary_meaning",
                            "latin"),
                    )),
                target_deck="Latin")
            for card in settings.cards)
        settings = replace(
            settings,
            cards=cards,
            target_deck="Latin",
            share_field_settings=False)
        pipeline = pipeline_store.replace_active_language_settings(
            pipeline,
            settings,
            (settings,))

        with tempfile.TemporaryDirectory() as directory:
            pipeline_runner.run_pipelines(
                "floreo",
                (pipeline,),
                project_root=PROJECT_ROOT,
                output_root=directory,
                generator=self.generator,
                importer=self.importer)

        self.generator.assert_called_once()
        prompt = self.generator.call_args.kwargs["prompt_text"]
        self.assertEqual(
            prompt.count('For "Translation (English)"'),
            1)
        self.assertEqual(
            prompt.count('For "Dictionary Meaning (Latin)"'),
            1)
        self.assertNotIn("Pronunciation", prompt)

    def test_separate_decks_are_passed_as_model_routes(self):
        pipeline = configure_pipeline(
            pipeline_store.default_pipeline(),
            "english",
            enabled=("word_to_meaning", "meaning_to_word"),
            separate=True)

        with tempfile.TemporaryDirectory() as directory:
            pipeline_runner.run_pipelines(
                "astrolabe",
                (pipeline,),
                project_root=PROJECT_ROOT,
                output_root=directory,
                generator=self.generator,
                importer=self.importer)

        routes = self.importer.call_args.kwargs["target_decks"]
        self.assertEqual(
            routes,
            (
                (
                    templates.get_direction_card_type(
                        "english",
                        "word_to_meaning").model.name,
                    "Target::1",
                ),
                (
                    templates.get_direction_card_type(
                        "english",
                        "meaning_to_word").model.name,
                    "Target::2",
                ),
            ))

    def test_generation_failure_does_not_stop_next_pipeline(self):
        first = pipeline_store.default_pipeline()
        second = pipeline_store.create_pipeline((first,))
        self.generator.side_effect = [
            RuntimeError("first failed"),
            Path("/tmp/second.apkg"),
        ]

        summary = pipeline_runner.run_pipelines(
            "words",
            (first, second),
            project_root=PROJECT_ROOT,
            generator=self.generator,
            importer=self.importer)

        self.assertEqual(len(summary.failures), 1)
        self.assertEqual(summary.failures[0].stage, "generation")
        self.assertEqual(len(summary.successes), 1)
        self.importer.assert_called_once()

    def test_missing_component_is_configuration_failure_without_api_call(self):
        pipeline = pipeline_store.default_pipeline()

        with tempfile.TemporaryDirectory() as directory:
            summary = pipeline_runner.run_pipelines(
                "words",
                (pipeline,),
                project_root=directory,
                generator=self.generator,
                importer=self.importer)

        self.assertEqual(len(summary.failures), 1)
        self.assertEqual(
            summary.failures[0].stage,
            "configuration")
        self.generator.assert_not_called()
        self.importer.assert_not_called()

    def test_import_failure_keeps_generated_output_in_summary(self):
        pipeline = pipeline_store.default_pipeline()
        self.importer.side_effect = RuntimeError("Anki unavailable")

        with tempfile.TemporaryDirectory() as directory:
            summary = pipeline_runner.run_pipelines(
                "words",
                (pipeline,),
                project_root=PROJECT_ROOT,
                output_root=directory,
                generator=self.generator,
                importer=self.importer)

        self.assertEqual(len(summary.failures), 1)
        failure = summary.failures[0]
        self.assertEqual(failure.stage, "import")
        self.assertIsNotNone(failure.output_path)

    def test_default_and_named_pipeline_output_paths_are_isolated(self):
        first = pipeline_store.default_pipeline()
        second = pipeline_store.create_pipeline((first,))
        root = Path("/temporary/output")

        first_paths = pipeline_runner.get_pipeline_output_paths(
            first,
            root)
        second_paths = pipeline_runner.get_pipeline_output_paths(
            second,
            root)

        self.assertEqual(
            first_paths["package"],
            root / "output.apkg")
        self.assertEqual(
            second_paths["package"],
            root / "pipelines" / second.pipeline_id / "output.apkg")


if __name__ == "__main__":
    unittest.main()
