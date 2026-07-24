import json
import os
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import genanki


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gui
import pipeline_store
import process_text
import templates


def configured_pipeline(
        language_key="english",
        *,
        enabled=("context",),
        shared_fields=None,
        share=True,
        per_card_fields=None):
    pipeline = pipeline_store.default_pipeline()
    settings = pipeline_store.get_language_settings(
        pipeline,
        language_key)
    shared_fields = (
        tuple(shared_fields)
        if shared_fields is not None
        else settings.shared_fields)
    per_card_fields = per_card_fields or {}
    cards = tuple(
        replace(
            card,
            enabled=card.direction_key in enabled,
            fields=tuple(per_card_fields.get(
                card.direction_key,
                card.fields)))
        for card in settings.cards)
    settings = replace(
        settings,
        cards=cards,
        share_field_settings=share,
        shared_fields=shared_fields)
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key=language_key)


def valid_default_card():
    return {
        "Word": "astrolabe",
        "Sentences": "One|Two|Three|Four",
        "Dictionary Meaning (English)": (
            "An instrument formerly used to determine celestial positions."),
        "Pronunciation (English)": "/ˈæstrəleɪb/",
    }


class FetchResponseTests(unittest.TestCase):
    def test_get_api_key_prefers_environment_variable(self):
        with (
                patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "environment-key"},
                    clear=True),
                patch.object(
                    process_text.credential_store,
                    "load_api_key") as load_api_key):
            result = process_text.get_api_key()

        self.assertEqual(result, "environment-key")
        load_api_key.assert_not_called()

    def test_get_api_key_falls_back_to_saved_key(self):
        with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    process_text.credential_store,
                    "load_api_key",
                    return_value="saved-key")):
            self.assertEqual(process_text.get_api_key(), "saved-key")

    def test_missing_api_key_fails_before_constructing_client(self):
        with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    process_text.credential_store,
                    "load_api_key",
                    return_value=None),
                patch.object(process_text, "OpenAI") as openai_class,
                self.assertRaises(process_text.MissingAPIKeyError)):
            process_text.fetch_response(
                "word",
                prompt_text="Prompt",
                response_path="/tmp/unused-response.json",
                response_log_path="/tmp/unused-response.log")

        openai_class.assert_not_called()

    def test_client_is_constructed_with_sdk_retries_disabled(self):
        fake_client = MagicMock()
        response = fake_client.responses.create.return_value
        response.status = "completed"
        response.output = []
        response.output_text = '{"cards": []}'

        with tempfile.TemporaryDirectory() as directory:
            with (
                    patch.dict(
                        os.environ,
                        {"OPENAI_API_KEY": "test-key"},
                        clear=True),
                    patch.object(
                        process_text,
                        "OpenAI",
                        return_value=fake_client) as openai_class):
                process_text.fetch_response(
                    "word",
                    prompt_text="Prompt: ",
                    response_path=Path(directory) / "response.json",
                    response_log_path=Path(directory) / "response.log")

        openai_class.assert_called_once_with(
            api_key="test-key",
            max_retries=0)
        fake_client.responses.create.assert_called_once()

    def test_fake_client_receives_schema_and_outputs_are_written(self):
        result = '{"cards": []}'
        client = MagicMock()
        response = client.responses.create.return_value
        response.status = "completed"
        response.output = []
        response.output_text = result
        pipeline = configured_pipeline(
            "french",
            enabled=("word_to_meaning",),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "translation",
                    "english"),
            ))

        with tempfile.TemporaryDirectory() as directory:
            response_path = Path(directory) / "response.json"
            log_path = Path(directory) / "response.log"
            returned = process_text.fetch_response(
                "épanouir",
                client=client,
                prompt_text="Define: ",
                response_path=response_path,
                response_log_path=log_path,
                response_format=process_text.build_response_format(
                    pipeline))

            self.assertEqual(returned, result)
            client.responses.create.assert_called_once_with(
                model="gpt-5.4-mini",
                input="Define: épanouir",
                text={"format": process_text.build_response_format(
                    pipeline)})
            self.assertEqual(
                response_path.read_text(encoding="utf-8"),
                result)
            self.assertEqual(
                log_path.read_text(encoding="utf-8"),
                "\n<break>\n" + result)

    def test_refusal_does_not_write_outputs(self):
        client = MagicMock()
        response = client.responses.create.return_value
        response.status = "completed"
        response.output = [{
            "type": "message",
            "content": [{
                "type": "refusal",
                "refusal": "Cannot comply.",
            }],
        }]
        response.output_text = ""

        with tempfile.TemporaryDirectory() as directory:
            response_path = Path(directory) / "response.json"
            log_path = Path(directory) / "response.log"
            with self.assertRaises(process_text.OpenAIRefusalError):
                process_text.fetch_response(
                    "word",
                    client=client,
                    prompt_text="Prompt",
                    response_path=response_path,
                    response_log_path=log_path)
            self.assertFalse(response_path.exists())
            self.assertFalse(log_path.exists())

    def test_incomplete_response_does_not_write_outputs(self):
        client = MagicMock()
        response = client.responses.create.return_value
        response.status = "incomplete"
        response.output = []
        response.incomplete_details.reason = "max_output_tokens"

        with tempfile.TemporaryDirectory() as directory:
            response_path = Path(directory) / "response.json"
            with self.assertRaisesRegex(
                    process_text.OpenAIIncompleteResponseError,
                    "max_output_tokens"):
                process_text.fetch_response(
                    "word",
                    client=client,
                    prompt_text="Prompt",
                    response_path=response_path,
                    response_log_path=Path(directory) / "log")
            self.assertFalse(response_path.exists())

    def test_fetch_response_file_reads_input_without_api_when_client_mocked(self):
        with tempfile.TemporaryDirectory() as directory:
            words_path = Path(directory) / "words"
            words_path.write_text("one\ntwo", encoding="utf-8")
            with patch.object(
                    process_text,
                    "fetch_response",
                    return_value="response") as fetch_response:
                result = process_text.fetch_response_file(
                    words_path,
                    client="fake-client")

        self.assertEqual(result, "response")
        fetch_response.assert_called_once_with(
            "one\ntwo",
            client="fake-client")


class ResponseSchemaTests(unittest.TestCase):
    def test_classical_chinese_era_uses_shared_term_field(self):
        pipeline = configured_pipeline(
            "classical_chinese_warring_states",
            enabled=("word_to_meaning",),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "english"),
            ))

        self.assertEqual(
            process_text.get_response_field_names(pipeline),
            (
                "Classical Chinese",
                "Dictionary Meaning (English)",
            ))
        self.assertEqual(
            process_text.build_response_format(pipeline)["name"],
            "autoanki_classical_chinese_warring_states_simple_cards")

    def test_schema_contains_only_selected_fields(self):
        pipeline = configured_pipeline(
            "french",
            enabled=("word_to_meaning", "meaning_to_word"),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "translation",
                    "english"),
                pipeline_store.FieldSetting(
                    "nuance",
                    "japanese"),
            ))
        response_format = process_text.build_response_format(pipeline)
        item = (
            response_format["schema"]["properties"]["cards"]["items"])

        self.assertEqual(
            response_format["name"],
            "autoanki_french_simple_cards")
        self.assertEqual(
            item["required"],
            [
                "French",
                "Translation (English)",
                "Nuance (Japanese)",
            ])
        self.assertEqual(
            set(item["properties"]),
            set(item["required"]))
        self.assertFalse(item["additionalProperties"])
        self.assertNotIn("Sentences", item["properties"])
        self.assertNotIn(
            "Dictionary Meaning (English)",
            item["properties"])
        self.assertNotIn("null", json.dumps(response_format))

    def test_context_adds_sentences_but_no_unselected_content(self):
        pipeline = configured_pipeline(
            "latin",
            enabled=("context",),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "latin"),
            ))
        required = process_text.build_response_format(
            pipeline)["schema"]["properties"]["cards"]["items"]["required"]

        self.assertEqual(
            required,
            [
                "Latin",
                "Sentences",
                "Dictionary Meaning (Latin)",
            ])

    def test_per_card_fields_are_deduplicated_by_field_and_language(self):
        translation = pipeline_store.FieldSetting(
            "translation",
            "english")
        pipeline = configured_pipeline(
            "japanese",
            enabled=("word_to_meaning", "meaning_to_word"),
            share=False,
            per_card_fields={
                "word_to_meaning": (translation,),
                "meaning_to_word": (
                    translation,
                    pipeline_store.FieldSetting(
                        "register",
                        "japanese"),
                ),
            })
        required = process_text.get_response_field_names(pipeline)

        self.assertEqual(
            required,
            (
                "Japanese",
                "Translation (English)",
                "Register (Japanese)",
            ))


class ProcessJsonTests(unittest.TestCase):
    def test_classical_chinese_era_uses_existing_classical_chinese_model(self):
        pipeline = configured_pipeline(
            "classical_chinese_ming",
            enabled=("word_to_meaning",),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "english"),
            ))
        deck = MagicMock()
        package = MagicMock()

        with (
                patch.object(process_text.genanki, "Note") as note_class,
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            process_text.process_json_text(
                json.dumps({"cards": [{
                    "Classical Chinese": "雅",
                    "Dictionary Meaning (English)": (
                        "Refined or elegant in a literary sense."),
                }]}),
                deck=deck,
                output_path="/tmp/ming.apkg",
                pipeline=pipeline)

        self.assertIs(
            note_class.call_args.kwargs["model"],
            templates.get_direction_card_type(
                "classical_chinese",
                "word_to_meaning").model)

    def test_one_response_creates_all_selected_directions(self):
        pipeline = configured_pipeline(
            "english",
            enabled=(
                "context",
                "word_to_meaning",
                "meaning_to_word"))
        deck = MagicMock()
        package = MagicMock()

        with (
                patch.object(process_text.genanki, "Note") as note_class,
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            count = process_text.process_json_text(
                json.dumps({"cards": [valid_default_card()]}),
                deck=deck,
                output_path="/tmp/multiple.apkg",
                pipeline=pipeline,
                guid_seed="pipeline-one")

        self.assertEqual(count, 3)
        self.assertEqual(note_class.call_count, 3)
        self.assertEqual(
            [
                item.kwargs["model"].name
                for item in note_class.call_args_list
            ],
            [
                templates.get_direction_card_type(
                    "english",
                    "context").model.name,
                templates.get_direction_card_type(
                    "english",
                    "word_to_meaning").model.name,
                templates.get_direction_card_type(
                    "english",
                    "meaning_to_word").model.name,
            ])
        self.assertEqual(deck.add_note.call_count, 3)
        package.write_to_file.assert_called_once_with(
            Path("/tmp/multiple.apkg"))

    def test_per_card_target_languages_fill_only_applicable_model_fields(self):
        translation = pipeline_store.FieldSetting(
            "translation",
            "english")
        french_nuance = pipeline_store.FieldSetting(
            "nuance",
            "french")
        japanese_dictionary = pipeline_store.FieldSetting(
            "dictionary_meaning",
            "japanese")
        pipeline = configured_pipeline(
            "french",
            enabled=("context", "word_to_meaning"),
            share=False,
            per_card_fields={
                "context": (translation, french_nuance),
                "word_to_meaning": (japanese_dictionary,),
            })
        response = {
            "French": "épanouir",
            "Sentences": "Une|Deux|Trois|Quatre",
            "Translation (English)": "to flourish",
            "Nuance (French)": "évoque un développement positif",
            "Dictionary Meaning (Japanese)": "十分に発達すること",
        }
        deck = MagicMock()
        package = MagicMock()

        with (
                patch.object(process_text.genanki, "Note") as note_class,
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            process_text.process_json_text(
                json.dumps({"cards": [response]}),
                deck=deck,
                output_path="/tmp/french.apkg",
                pipeline=pipeline)

        context_fields = note_class.call_args_list[0].kwargs["fields"]
        word_fields = note_class.call_args_list[1].kwargs["fields"]
        self.assertEqual(
            context_fields,
            [
                "épanouir",
                "Une|Deux|Trois|Quatre",
                "to flourish",
                "",
                "",
                "",
                "",
                "évoque un développement positif",
            ])
        self.assertEqual(
            word_fields,
            [
                "épanouir",
                "Une|Deux|Trois|Quatre",
                "",
                "十分に発達すること",
                "",
                "",
                "",
                "",
            ])

    def test_unselected_json_field_is_rejected_before_export(self):
        card = valid_default_card()
        card["Register (English)"] = "formal"
        deck = MagicMock()

        with (
                patch.object(process_text.genanki, "Package") as package,
                self.assertRaisesRegex(
                    process_text.GeneratedCardValidationError,
                    "unexpected Register")):
            process_text.process_json_text(
                json.dumps({"cards": [card]}),
                deck=deck)

        package.assert_not_called()
        deck.add_note.assert_not_called()

    def test_missing_selected_json_field_is_rejected(self):
        card = valid_default_card()
        del card["Pronunciation (English)"]

        with self.assertRaisesRegex(
                process_text.GeneratedCardValidationError,
                "missing Pronunciation"):
            process_text.process_json_text(
                json.dumps({"cards": [card]}),
                deck=MagicMock())

    def test_empty_or_non_string_selected_field_is_rejected(self):
        card = valid_default_card()
        card["Pronunciation (English)"] = ""
        with self.assertRaisesRegex(
                process_text.GeneratedCardValidationError,
                "cannot be empty"):
            process_text.process_json_text(
                json.dumps({"cards": [card]}),
                deck=MagicMock())

        card["Pronunciation (English)"] = None
        with self.assertRaises(TypeError):
            process_text.process_json_text(
                json.dumps({"cards": [card]}),
                deck=MagicMock())

    def test_empty_nuance_is_accepted_and_hidden_by_the_card_template(self):
        pipeline = configured_pipeline(
            "french",
            enabled=("word_to_meaning",),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "english"),
                pipeline_store.FieldSetting(
                    "nuance",
                    "english"),
            ))
        response = {
            "French": "table",
            "Dictionary Meaning (English)": (
                "A piece of furniture with a flat top."),
            "Nuance (English)": "   ",
        }
        deck = MagicMock()
        package = MagicMock()

        with (
                patch.object(process_text.genanki, "Note") as note_class,
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            count = process_text.process_json_text(
                json.dumps({"cards": [response]}),
                deck=deck,
                output_path="/tmp/no-nuance.apkg",
                pipeline=pipeline)

        self.assertEqual(count, 1)
        self.assertEqual(
            note_class.call_args.kwargs["fields"][-1],
            "")
        answer = note_class.call_args.kwargs[
            "model"].templates[0]["afmt"]
        self.assertIn("{{#Nuance}}", answer)

    def test_empty_nuance_cannot_be_the_only_meaning_content(self):
        pipeline = configured_pipeline(
            "french",
            enabled=("meaning_to_word",),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "nuance",
                    "english"),
            ))

        with self.assertRaisesRegex(
                process_text.GeneratedCardValidationError,
                "no non-empty meaning fields"):
            process_text.process_json_text(
                json.dumps({"cards": [{
                    "French": "table",
                    "Nuance (English)": "",
                }]}),
                deck=MagicMock(),
                pipeline=pipeline)

    def test_wrong_sentence_count_is_rejected_before_export(self):
        card = valid_default_card()
        card["Sentences"] = "One|Two|Three"
        deck = MagicMock()

        with (
                patch.object(process_text.genanki, "Package") as package,
                self.assertRaisesRegex(
                    process_text.GeneratedCardValidationError,
                    "exactly four")):
            process_text.process_json_text(
                json.dumps({"cards": [card]}),
                deck=deck)

        package.assert_not_called()
        deck.add_note.assert_not_called()

    def test_invalid_root_shapes_are_rejected(self):
        for data in (
                {"cards": [], "extra": True},
                {"not_cards": []},
                {"cards": "not an array"},
                [None]):
            with self.subTest(data=data):
                with self.assertRaises(
                        (process_text.GeneratedCardValidationError,
                         TypeError)):
                    process_text.process_json_text(
                        json.dumps(data),
                        deck=MagicMock())

    def test_bare_legacy_array_root_is_still_accepted(self):
        deck = MagicMock()
        package = MagicMock()
        with patch.object(
                process_text.genanki,
                "Package",
                return_value=package):
            count = process_text.process_json_text(
                json.dumps([valid_default_card()]),
                deck=deck,
                output_path="/tmp/legacy-array.apkg")

        self.assertEqual(count, 1)
        deck.add_note.assert_called_once()

    def test_guid_is_stable_and_scoped_to_pipeline_and_direction(self):
        pipeline = configured_pipeline(
            "english",
            enabled=("word_to_meaning",))
        card = valid_default_card()
        del card["Sentences"]
        deck = MagicMock()
        package = MagicMock()
        with (
                patch.object(process_text.genanki, "Note") as note_class,
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            process_text.process_json_text(
                json.dumps({"cards": [card]}),
                deck=deck,
                output_path="/tmp/guid.apkg",
                pipeline=pipeline,
                guid_seed="pipeline-one")

        expected = process_text.genanki.guid_for(
            "autoanki",
            "pipeline-one",
            "english_word_to_meaning",
            "astrolabe",
            (
                "An instrument formerly used to determine "
                "celestial positions."),
            "/ˈæstrəleɪb/")
        self.assertEqual(
            note_class.call_args.kwargs["guid"],
            expected)
        self.assertNotEqual(
            expected,
            process_text.genanki.guid_for(
                "autoanki",
                "pipeline-two",
                "english_word_to_meaning",
                "astrolabe",
                (
                    "An instrument formerly used to determine "
                    "celestial positions."),
                "/ˈæstrəleɪb/"))

    def test_real_anki_package_is_created(self):
        deck = genanki.Deck(
            2059400111,
            "Test Generated Words")

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "output.apkg"
            process_text.process_json_text(
                json.dumps({"cards": [valid_default_card()]}),
                deck=deck,
                output_path=output_path)
            self.assertTrue(output_path.is_file())
            with zipfile.ZipFile(output_path) as package:
                self.assertIn(
                    "collection.anki2",
                    package.namelist())
                self.assertIn("media", package.namelist())

    def test_process_json_file_reads_supplied_file(self):
        with tempfile.TemporaryDirectory() as directory:
            response_path = Path(directory) / "response.json"
            response_path.write_text("[]", encoding="utf-8")
            with patch.object(
                    process_text,
                    "process_json_text") as process:
                process_text.process_json_file(
                    response_path,
                    output_path="deck.apkg")

        process.assert_called_once_with(
            "[]",
            output_path="deck.apkg")


class GenerateDeckTests(unittest.TestCase):
    def test_generate_deck_uses_fresh_deck_and_composed_pipeline(self):
        pipeline = configured_pipeline(
            "classical_chinese",
            enabled=("meaning_to_word",),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "classical_chinese"),
            ))
        fresh_deck = MagicMock()

        with (
                patch.object(
                    process_text,
                    "fetch_response",
                    return_value='{"cards": []}') as fetch,
                patch.object(
                    process_text.prompt_builder,
                    "build_prompt",
                    return_value="composed prompt") as build_prompt,
                patch.object(
                    process_text.templates,
                    "create_deck",
                    return_value=fresh_deck) as create_deck,
                patch.object(
                    process_text,
                    "process_json_text") as process):
            result = process_text.generate_deck(
                "學",
                client="fake-client",
                output_path="custom.apkg",
                pipeline=pipeline)

        self.assertEqual(result, Path("custom.apkg"))
        build_prompt.assert_called_once_with(pipeline)
        fetch.assert_called_once_with(
            "學",
            client="fake-client",
            prompt_path=None,
            prompt_text="composed prompt",
            response_path=process_text.RESPONSE_PATH,
            response_log_path=process_text.RESPONSE_LOG_PATH,
            response_format=process_text.build_response_format(pipeline))
        create_deck.assert_called_once_with(
            templates.DECK_ID,
            templates.DECK_NAME)
        process.assert_called_once_with(
            '{"cards": []}',
            deck=fresh_deck,
            output_path=Path("custom.apkg"),
            pipeline=pipeline,
            guid_seed=None)

    def test_create_deck_returns_new_empty_deck(self):
        first = templates.create_deck()
        second = templates.create_deck()

        self.assertIsNot(first, second)
        self.assertEqual(first.notes, [])
        self.assertEqual(second.notes, [])


class GuiLogicTests(unittest.TestCase):
    def test_language_selectors_fit_the_longest_available_label(self):
        longest_label = max(
            (
                language.name
                for language in pipeline_store.list_languages()
            ),
            key=len)

        self.assertGreater(
            gui.language_selector_width(),
            len(longest_label))
        self.assertEqual(
            longest_label,
            "Classical Chinese (Warring States)")

    def test_generate_language_dropdown_resolves_explicit_selection(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.generation_language = MagicMock(
            get=MagicMock(return_value="Japanese"))

        language = app.get_generation_language()

        self.assertEqual(language.key, "japanese")

    def test_viewing_another_settings_tab_does_not_change_generation_language(
            self):
        editor = object.__new__(gui.PipelineEditor)
        editor.last_active_language_key = "english"
        editor.get_active_language = MagicMock(
            return_value=pipeline_store.get_language("french"))
        editor._update_visibility = MagicMock()
        editor.app = MagicMock()
        editor.app.generation_language.get.return_value = "English"

        editor._language_changed()

        self.assertEqual(
            editor.app.generation_language.get(),
            "English")
        editor._update_visibility.assert_called_once_with(
            "french")
        editor.app.schedule_pipeline_save.assert_not_called()

    def test_translation_language_menu_excludes_source_only(self):
        editor = object.__new__(gui.PipelineEditor)

        translation = editor._language_values(
            "french",
            "translation")
        dictionary = editor._language_values(
            "french",
            "dictionary_meaning")

        self.assertNotIn("French", translation)
        self.assertIn("English", translation)
        self.assertIn("French", dictionary)
        self.assertEqual(
            len(dictionary),
            len(pipeline_store.list_response_languages()))

    def test_visibility_uses_scrollable_layout_for_large_controls(self):
        editor = object.__new__(gui.PipelineEditor)
        editor._refresh_layout_geometry = MagicMock()
        editor.separate_target_deck_variables = {
            "english": MagicMock(get=MagicMock(return_value=True))}
        editor.share_field_settings_variables = {
            "english": MagicMock(get=MagicMock(return_value=False))}
        editor.target_deck_boxes = {"english": MagicMock()}
        editor.shared_field_containers = {"english": MagicMock()}
        editor.direction_enabled_variables = {
            "english": {
                direction.key: MagicMock(
                    get=MagicMock(return_value=True))
                for direction in pipeline_store.list_directions()
            }}
        editor.direction_target_deck_boxes = {
            "english": {
                direction.key: MagicMock()
                for direction in pipeline_store.list_directions()
            }}
        editor.per_card_field_containers = {
            "english": {
                direction.key: MagicMock()
                for direction in pipeline_store.list_directions()
            }}
        editor.shared_field_language_boxes = {
            "english": {
                field.key: MagicMock()
                for field in pipeline_store.list_field_options()
            }}
        editor.shared_field_enabled_variables = {
            "english": {
                field.key: MagicMock(
                    get=MagicMock(return_value=True))
                for field in pipeline_store.list_field_options()
            }}
        editor.per_card_field_language_boxes = {
            "english": {
                direction.key: {
                    field.key: MagicMock()
                    for field in pipeline_store.list_field_options()
                }
                for direction in pipeline_store.list_directions()
            }}
        editor.per_card_field_enabled_variables = {
            "english": {
                direction.key: {
                    field.key: MagicMock(
                        get=MagicMock(return_value=True))
                    for field in pipeline_store.list_field_options()
                }
                for direction in pipeline_store.list_directions()
            }}

        editor._update_visibility("english")

        editor.target_deck_boxes[
            "english"].configure.assert_called_with(
                state="disabled")
        editor.shared_field_containers[
            "english"].grid_remove.assert_called_once()
        for direction in pipeline_store.list_directions():
            editor.direction_target_deck_boxes[
                "english"][direction.key].configure.assert_called_with(
                    state="normal")
            editor.per_card_field_containers[
                "english"][direction.key].grid.assert_called_once()
        editor._refresh_layout_geometry.assert_called_once_with(
            "english")

    def test_field_language_menus_remain_available_when_field_is_unticked(
            self):
        editor = object.__new__(gui.PipelineEditor)
        shared_boxes = {
            field.key: MagicMock()
            for field in pipeline_store.list_field_options()
        }
        per_card_boxes = {
            direction.key: {
                field.key: MagicMock()
                for field in pipeline_store.list_field_options()
            }
            for direction in pipeline_store.list_directions()
        }
        editor.shared_field_language_boxes = {
            "english": shared_boxes}
        editor.per_card_field_language_boxes = {
            "english": per_card_boxes}

        editor._update_field_box_states("english")

        for box in shared_boxes.values():
            box.configure.assert_called_once_with(
                state="readonly")
        for boxes in per_card_boxes.values():
            for box in boxes.values():
                box.configure.assert_called_once_with(
                    state="readonly")

    def test_unticked_field_language_is_remembered_but_not_generated(self):
        editor = object.__new__(gui.PipelineEditor)
        enabled = {
            field.key: MagicMock(
                get=MagicMock(
                    return_value=field.key == "nuance"))
            for field in pipeline_store.list_field_options()
        }
        languages = {
            field.key: MagicMock(
                get=MagicMock(
                    return_value=(
                        "Japanese"
                        if field.key == "translation"
                        else "Latin")))
            for field in pipeline_store.list_field_options()
        }

        fields = editor._fields_from_controls(
            enabled,
            languages)

        self.assertNotIn(
            pipeline_store.FieldSetting(
                "translation",
                "japanese"),
            fields)
        self.assertIn(
            pipeline_store.FieldSetting("nuance", "latin"),
            fields)
        self.assertEqual(
            languages["translation"].get(),
            "Japanese")

    def test_initial_window_size_respects_screen_and_large_default(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.root.winfo_screenwidth.return_value = 1920
        app.root.winfo_screenheight.return_value = 1080

        app._set_initial_window_size()

        app.root.geometry.assert_called_once_with(
            "1840x980+40+50")
        app.root.minsize.assert_called_once_with(1000, 720)

    def test_wheel_events_are_normalized(self):
        self.assertEqual(
            gui.wheel_scroll_amount(
                MagicMock(num=4, state=0)),
            -1)
        self.assertEqual(
            gui.wheel_scroll_amount(
                MagicMock(num=5, state=0)),
            1)
        self.assertEqual(
            gui.wheel_scroll_amount(
                MagicMock(num=None, delta=120, state=0)),
            -1)


class EntrypointTests(unittest.TestCase):
    def test_main_reads_words_then_generates_deck(self):
        with (
                patch.object(
                    type(process_text.WORDS_PATH),
                    "read_text",
                    return_value="one\ntwo") as read_text,
                patch.object(
                    process_text,
                    "generate_deck") as generate):
            process_text.main()

        read_text.assert_called_once_with(encoding="utf-8")
        generate.assert_called_once_with("one\ntwo")


if __name__ == "__main__":
    unittest.main()
