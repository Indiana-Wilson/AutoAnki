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


class PipelineRunnerTests(unittest.TestCase):
    def _project_with_prompts(self, directory):
        project_root = Path(directory)
        input_directory = project_root / "input"
        prompt_directory = input_directory / "prompts"
        prompt_directory.mkdir(parents=True)
        (prompt_directory / "english_vocab").write_text(
            "default prompt",
            encoding="utf-8")
        (prompt_directory / "english_vocab_simple").write_text(
            "simple English prompt",
            encoding="utf-8")
        (prompt_directory / "classical_chinese").write_text(
            "Classical Chinese prompt",
            encoding="utf-8")
        (prompt_directory / "classical_chinese_simple").write_text(
            "simple Classical Chinese prompt",
            encoding="utf-8")
        (prompt_directory / "french_vocab").write_text(
            "French prompt",
            encoding="utf-8")
        (prompt_directory / "french_vocab_simple").write_text(
            "simple French prompt",
            encoding="utf-8")
        (prompt_directory / "japanese_vocab").write_text(
            "Japanese prompt",
            encoding="utf-8")
        (prompt_directory / "japanese_vocab_simple").write_text(
            "simple Japanese prompt",
            encoding="utf-8")
        (prompt_directory / "latin_vocab").write_text(
            "Latin prompt",
            encoding="utf-8")
        (prompt_directory / "latin_vocab_simple").write_text(
            "simple Latin prompt",
            encoding="utf-8")
        return project_root

    def test_two_pipelines_generate_and_import_independently(self):
        first = pipeline_store.default_pipeline()
        second = replace(
            pipeline_store.create_pipeline((first,)),
            language_key="classical_chinese",
            card_type_keys=(templates.CLASSICAL_CHINESE_CARD_TYPE.key,),
            target_deck="Another::Deck")
        generator = MagicMock(
            side_effect=lambda _words, **kwargs: kwargs["output_path"])
        importer = MagicMock(
            side_effect=lambda _path, **kwargs: (
                anki_integration.AnkiImportResult(
                    cards_moved=2,
                    target_deck=kwargs["target_deck"])))
        progress = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = self._project_with_prompts(
                temporary_directory)
            summary = pipeline_runner.run_pipelines(
                "astrolabe",
                (first, second),
                project_root=project_root,
                output_root=project_root / "output",
                progress_callback=progress.append,
                openai_client="openai-client",
                anki_client="anki-client",
                generator=generator,
                importer=importer)

        self.assertEqual(len(summary.successes), 2)
        self.assertEqual(summary.cards_moved, 4)
        self.assertEqual(
            [item.stage for item in progress],
            ["generating", "importing", "generating", "importing"])
        self.assertEqual(generator.call_count, 2)
        self.assertEqual(importer.call_count, 2)

        first_generation = generator.call_args_list[0].kwargs
        second_generation = generator.call_args_list[1].kwargs
        self.assertEqual(
            first_generation["guid_seed"],
            first.pipeline_id)
        self.assertEqual(
            first_generation["legacy_guid_card_type_key"],
            templates.DEFAULT_CARD_TYPE_KEY)
        self.assertEqual(
            second_generation["guid_seed"],
            second.pipeline_id)
        self.assertNotEqual(
            first_generation["deck_id"],
            second_generation["deck_id"])
        self.assertEqual(
            importer.call_args_list[1].kwargs["target_deck"],
            "Another::Deck")

    def test_failure_in_one_pipeline_does_not_stop_the_next(self):
        first = pipeline_store.default_pipeline()
        second = pipeline_store.create_pipeline((first,))
        generator = MagicMock(
            side_effect=[
                RuntimeError("first failed"),
                Path("/tmp/second.apkg"),
            ])
        importer = MagicMock(return_value=(
            anki_integration.AnkiImportResult(
                cards_moved=1,
                target_deck=second.target_deck)))

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = self._project_with_prompts(
                temporary_directory)
            summary = pipeline_runner.run_pipelines(
                "words",
                (first, second),
                project_root=project_root,
                generator=generator,
                importer=importer)

        self.assertEqual(len(summary.failures), 1)
        self.assertEqual(
            summary.failures[0].stage,
            "generation")
        self.assertEqual(len(summary.successes), 1)
        importer.assert_called_once()

    def test_classical_chinese_pipeline_combines_prompt_model_and_target(self):
        pipeline = replace(
            pipeline_store.create_pipeline(),
            language_key="classical_chinese",
            card_type_keys=(
                templates.CLASSICAL_CHINESE_CARD_TYPE.key,),
            target_deck="Retained Information::Chinese::Classical Chinese")
        generator = MagicMock(
            side_effect=lambda _words, **kwargs: kwargs["output_path"])
        importer = MagicMock(return_value=(
            anki_integration.AnkiImportResult(
                cards_moved=1,
                target_deck=pipeline.target_deck)))

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = self._project_with_prompts(temporary_directory)
            summary = pipeline_runner.run_pipelines(
                "學",
                (pipeline,),
                project_root=project_root,
                generator=generator,
                importer=importer)

        self.assertEqual(len(summary.successes), 1)
        generation = generator.call_args.kwargs
        self.assertEqual(
            generation["prompt_path"].name,
            "classical_chinese")
        self.assertEqual(
            generation["card_type_keys"],
            ("classical_chinese_vocabulary",))
        importer.assert_called_once_with(
            generation["output_path"],
            target_deck=pipeline.target_deck,
            source_deck=pipeline.generated_deck_name,
            client=None)

    def test_two_simple_outputs_share_the_minimal_prompt_and_request(self):
        pipeline = replace(
            pipeline_store.create_pipeline(),
            card_type_keys=(
                templates.ENGLISH_WORD_TO_MEANING_CARD_TYPE.key,
                templates.ENGLISH_MEANING_TO_WORD_CARD_TYPE.key))
        generator = MagicMock(
            side_effect=lambda _words, **kwargs: kwargs["output_path"])
        importer = MagicMock(return_value=(
            anki_integration.AnkiImportResult(
                cards_moved=2,
                target_deck=pipeline.target_deck)))

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = self._project_with_prompts(temporary_directory)
            summary = pipeline_runner.run_pipelines(
                "astrolabe",
                (pipeline,),
                project_root=project_root,
                generator=generator,
                importer=importer)

        self.assertEqual(len(summary.successes), 1)
        generator.assert_called_once()
        generation = generator.call_args.kwargs
        self.assertEqual(
            generation["prompt_path"].name,
            "english_vocab_simple")
        self.assertEqual(
            generation["card_type_keys"],
            (
                "english_word_to_meaning",
                "english_meaning_to_word"))

    def test_selected_outputs_can_be_imported_into_separate_decks(self):
        pipeline = replace(
            pipeline_store.create_pipeline(),
            card_type_keys=(
                templates.ENGLISH_WORD_TO_MEANING_CARD_TYPE.key,
                templates.ENGLISH_MEANING_TO_WORD_CARD_TYPE.key),
            separate_target_decks=True,
            card_type_target_decks=(
                (
                    templates.ENGLISH_WORD_TO_MEANING_CARD_TYPE.key,
                    "English::Recognition"),
                (
                    templates.ENGLISH_MEANING_TO_WORD_CARD_TYPE.key,
                    "English::Production"),
            ))
        generator = MagicMock(
            side_effect=lambda _words, **kwargs: kwargs["output_path"])
        importer = MagicMock(return_value=(
            anki_integration.AnkiImportResult(
                cards_moved=2,
                target_deck="Multiple decks")))

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = self._project_with_prompts(temporary_directory)
            pipeline_runner.run_pipelines(
                "astrolabe",
                (pipeline,),
                project_root=project_root,
                generator=generator,
                importer=importer)

        generator.assert_called_once()
        importer.assert_called_once_with(
            generator.call_args.kwargs["output_path"],
            target_deck=pipeline.target_deck,
            target_decks=(
                (
                    templates
                    .ENGLISH_WORD_TO_MEANING_CARD_TYPE.model.name,
                    "English::Recognition"),
                (
                    templates
                    .ENGLISH_MEANING_TO_WORD_CARD_TYPE.model.name,
                    "English::Production"),
            ),
            source_deck=pipeline.generated_deck_name,
            client=None)

    def test_mixed_japanese_definitions_share_one_detailed_request(self):
        pipeline = replace(
            pipeline_store.create_pipeline(),
            language_key="japanese",
            card_type_keys=(
                templates.JAPANESE_VOCABULARY_CARD_TYPE.key,
                templates.JAPANESE_WORD_TO_NATIVE_MEANING_CARD_TYPE.key),
            target_deck="Retained Information::Japanese")
        generator = MagicMock(
            side_effect=lambda _words, **kwargs: kwargs["output_path"])
        importer = MagicMock(return_value=(
            anki_integration.AnkiImportResult(
                cards_moved=2,
                target_deck=pipeline.target_deck)))

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = self._project_with_prompts(temporary_directory)
            summary = pipeline_runner.run_pipelines(
                "勉強",
                (pipeline,),
                project_root=project_root,
                generator=generator,
                importer=importer)

        self.assertEqual(len(summary.successes), 1)
        generator.assert_called_once()
        generation = generator.call_args.kwargs
        self.assertEqual(
            generation["prompt_path"].name,
            "japanese_vocab")
        self.assertEqual(
            generation["card_type_keys"],
            (
                "japanese_vocabulary",
                "japanese_word_to_native_meaning"))

    def test_latin_native_outputs_use_one_minimal_request(self):
        pipeline = replace(
            pipeline_store.create_pipeline(),
            language_key="latin",
            card_type_keys=(
                templates.LATIN_WORD_TO_NATIVE_MEANING_CARD_TYPE.key,
                templates.LATIN_NATIVE_MEANING_TO_WORD_CARD_TYPE.key),
            target_deck="Retained Information::Latin")
        generator = MagicMock(
            side_effect=lambda _words, **kwargs: kwargs["output_path"])
        importer = MagicMock(return_value=(
            anki_integration.AnkiImportResult(
                cards_moved=2,
                target_deck=pipeline.target_deck)))

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = self._project_with_prompts(temporary_directory)
            summary = pipeline_runner.run_pipelines(
                "sapientia",
                (pipeline,),
                project_root=project_root,
                generator=generator,
                importer=importer)

        self.assertEqual(len(summary.successes), 1)
        generator.assert_called_once()
        self.assertEqual(
            generator.call_args.kwargs["prompt_path"].name,
            "latin_vocab_simple")
        self.assertEqual(
            generator.call_args.kwargs["card_type_keys"],
            (
                "latin_word_to_native_meaning",
                "latin_native_meaning_to_word"))

    def test_default_pipeline_keeps_legacy_output_paths(self):
        paths = pipeline_runner.get_pipeline_output_paths(
            pipeline_store.default_pipeline(),
            Path("/project/output"))

        self.assertEqual(
            paths["response"],
            Path("/project/output/response.json"))
        self.assertEqual(
            paths["package"],
            Path("/project/output/output.apkg"))


if __name__ == "__main__":
    unittest.main()
