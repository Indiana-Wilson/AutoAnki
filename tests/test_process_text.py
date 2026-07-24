import json
import os
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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
                reasoning={"effort": "none"},
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
    def test_source_chunk_size_accepts_positive_integers_only(self):
        self.assertEqual(gui.parse_source_chunk_size("500"), 500)
        self.assertEqual(gui.parse_source_chunk_size(" 50 "), 50)

        for invalid in ("", "0", "-1", "2.5", "ten", "10001"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    gui.parse_source_chunk_size(invalid)

    def test_source_context_options_use_backend_canonical_keys(self):
        self.assertEqual(
            tuple(
                key
                for key, _label, _description
                in gui.SOURCE_CONTEXT_OPTIONS),
            (
                "none",
                "sentence",
                "sentence_neighbors",
                "chunk_span",
            ))

    def test_source_catalogue_normalisation_ignores_invalid_and_duplicates(
            self):
        options = gui.normalise_source_options((
            {
                "key": "prepared_one",
                "name": "Prepared One",
                "word_count": 12,
            },
            {
                "key": "prepared_one",
                "name": "Duplicate",
                "word_count": 20,
            },
            {
                "key": "",
                "name": "Missing Key",
            },
            {
                "key": "negative",
                "name": "Negative",
                "word_count": -1,
            },
        ))

        self.assertEqual(
            options,
            (
                gui.SourceUiOption(
                    key="prepared_one",
                    name="Prepared One",
                    word_count=12),
            ))

    def test_duplicate_source_titles_receive_stable_keyed_display_labels(self):
        first = gui.SourceUiOption(
            key="custom_alpha",
            name="Shared title")
        second = gui.SourceUiOption(
            key="custom_beta",
            name="Shared title")

        forward = gui.source_option_display_labels((first, second))
        reversed_order = gui.source_option_display_labels((second, first))

        self.assertEqual(forward, reversed_order)
        self.assertEqual(
            forward,
            {
                "custom_alpha": "Shared title · custom_alpha",
                "custom_beta": "Shared title · custom_beta",
            })
        self.assertEqual(first.name, second.name)

    def test_duplicate_source_display_labels_select_distinct_source_keys(self):
        app = object.__new__(gui.AutoAnkiApp)
        selected = {"label": ""}
        app.source_selected_label = MagicMock()
        app.source_selected_label.get.side_effect = (
            lambda: selected["label"])
        app.source_selected_label.set.side_effect = (
            lambda label: selected.__setitem__("label", label))
        app.source_options = ()
        app.source_options_by_label = {}
        app.source_options_by_key = {}
        app.source_display_labels_by_key = {}
        app.source_catalog_loader = MagicMock(return_value=(
            {
                "key": "custom_alpha",
                "name": "Shared title",
            },
            {
                "key": "custom_beta",
                "name": "Shared title",
            },
        ))
        app.source_selector = MagicMock()
        app.source_preview_selector = MagicMock()
        app.source_catalog_status = MagicMock()
        app._source_selection_changed = MagicMock()

        app._load_source_catalogue()

        labels = tuple(
            label
            for label, option in app.source_options_by_label.items()
            if option.name == "Shared title")
        self.assertEqual(len(labels), 2)
        selected["label"] = labels[0]
        first_key = app._selected_source_option().key
        selected["label"] = labels[1]
        second_key = app._selected_source_option().key
        self.assertEqual(
            {first_key, second_key},
            {"custom_alpha", "custom_beta"})
        self.assertTrue(all(
            app.source_options_by_label[label].name == "Shared title"
            for label in labels))

    def test_source_preview_contract_preserves_ordered_context_metadata(self):
        page = gui.normalise_source_preview_response({
            "source_key": "journey_to_the_west",
            "total": 24224,
            "offset": 500,
            "items": (
                {
                    "rank": 501,
                    "term": "行者",
                    "section_title": "第003回",
                    "previous_sentence": "前句。",
                    "current_sentence": "行者在此。",
                    "next_sentence": "後句。",
                },
                {
                    "rank": 502,
                    "term": "在此",
                    "section_title": "第003回",
                    "previous_sentence": None,
                    "current_sentence": "行者在此。",
                    "next_sentence": "",
                },
            ),
        })

        self.assertEqual(page.source_key, "journey_to_the_west")
        self.assertEqual(page.total, 24224)
        self.assertEqual(page.offset, 500)
        self.assertEqual(
            tuple(item.rank for item in page.items),
            (501, 502))
        self.assertEqual(page.items[0].section_title, "第003回")
        self.assertEqual(page.items[0].previous_sentence, "前句。")
        self.assertEqual(page.items[1].previous_sentence, "")

    def test_source_preview_contract_rejects_unordered_or_overflowing_pages(
            self):
        base = {
            "source_key": "fixture",
            "total": 2,
            "offset": 0,
            "items": (
                {
                    "rank": 2,
                    "term": "甲",
                    "section_title": "一",
                    "previous_sentence": "",
                    "current_sentence": "甲。",
                    "next_sentence": "",
                },
                {
                    "rank": 1,
                    "term": "乙",
                    "section_title": "一",
                    "previous_sentence": "",
                    "current_sentence": "乙。",
                    "next_sentence": "",
                },
            ),
        }
        with self.assertRaisesRegex(ValueError, "strictly ordered"):
            gui.normalise_source_preview_response(base)

        overflowing = dict(base)
        overflowing["offset"] = 2
        overflowing["items"] = base["items"][:1]
        with self.assertRaisesRegex(ValueError, "beyond"):
            gui.normalise_source_preview_response(overflowing)

    def test_source_preview_loader_runs_off_tk_thread_with_small_mapping(
            self):
        app = object.__new__(gui.AutoAnkiApp)
        app.source_preview_loader = MagicMock(return_value={
            "source_key": "fixture",
            "total": 0,
            "offset": 0,
            "items": (),
        })
        app._selected_source_option = MagicMock(
            return_value=gui.SourceUiOption(
                key="fixture",
                name="Fixture"))
        app.source_preview_page_size = MagicMock(
            get=MagicMock(return_value="500"))
        app.source_preview_generation = 0
        app.source_preview_pending = set()
        app.source_preview_polling = False
        app.source_preview_result_queue = gui.queue.Queue()
        app.source_preview_status = MagicMock()
        app._update_source_preview_navigation = MagicMock()
        app.root = MagicMock()

        with patch.object(gui.threading, "Thread") as thread_class:
            worker = thread_class.return_value
            app._request_source_preview(offset=0)

            app.source_preview_loader.assert_not_called()
            worker.start.assert_called_once_with()
            target = thread_class.call_args.kwargs["target"]
            target()

        app.source_preview_loader.assert_called_once_with({
            "source_key": "fixture",
            "offset": 0,
            "limit": 500,
        })
        generation, request, outcome, value = (
            app.source_preview_result_queue.get_nowait())
        self.assertEqual(generation, 1)
        self.assertEqual(
            request,
            {
                "source_key": "fixture",
                "offset": 0,
                "limit": 500,
            })
        self.assertEqual(outcome, "success")
        self.assertEqual(value["items"], ())

    def test_manual_input_filter_contract_uses_each_nonblank_line(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.get_generation_language = MagicMock(
            return_value=pipeline_store.get_language("french"))
        app._selected_anki_exclusions = MagicMock(return_value=(
            {
                "deck": "Known vocabulary",
                "model": "Vocabulary",
                "field": "Word",
            },
            {
                "deck": "Archive",
                "model": "Old Vocabulary",
                "field": "Term",
            },
        ))

        request = app._manual_input_filter_request(
            "  déjà vu  \n\n行者\nthird item  ")

        self.assertEqual(
            request,
            {
                "text": "  déjà vu  \n\n行者\nthird item  ",
                "candidates": ("déjà vu", "行者", "third item"),
                "anki_exclusions": (
                    {
                        "deck": "Known vocabulary",
                        "model": "Vocabulary",
                        "field": "Word",
                    },
                    {
                        "deck": "Archive",
                        "model": "Old Vocabulary",
                        "field": "Term",
                    },
                ),
            })
        app._selected_anki_exclusions.assert_called_once_with("french")

    def test_manual_input_filter_callback_starts_off_tk_thread(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.input_text = MagicMock()
        app.input_text.get.return_value = "known\nnew\n"
        app.save_pipeline_rows = MagicMock(return_value=("pipeline",))
        app.get_generation_language = MagicMock(
            return_value=pipeline_store.get_language("english"))
        app._language_filter = MagicMock(
            return_value=MagicMock(enabled=True))
        request = {
            "text": "known\nnew",
            "candidates": ("known", "new"),
            "anki_exclusion": {
                "deck": "Known",
                "model": "Basic",
                "card_template_name": None,
                "field": "Front",
            },
        }
        app._manual_input_filter_request = MagicMock(
            return_value=request)
        app.manual_input_filter_callback = MagicMock(return_value={
            "filtered_text": "new",
            "excluded_count": 1,
            "remaining_count": 1,
        })
        app._set_generation_busy = MagicMock()
        app._set_status = MagicMock()
        app.root = MagicMock()
        app.result_queue = gui.queue.Queue()

        with patch.object(gui.threading, "Thread") as thread_class:
            worker = thread_class.return_value
            app.start_generation()

            app.manual_input_filter_callback.assert_not_called()
            worker.start.assert_called_once_with()
            target = thread_class.call_args.kwargs["target"]
            args = thread_class.call_args.kwargs["args"]
            target(*args)

        app.manual_input_filter_callback.assert_called_once_with(request)
        app._language_filter.assert_called_once_with("english")
        outcome, value = app.result_queue.get_nowait()
        self.assertEqual(outcome, "manual_filter_complete")
        self.assertEqual(value[0].filtered_text, "new")

    def test_zero_remaining_manual_lines_never_start_a_pipeline(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.result_queue = gui.queue.Queue()
        app.result_queue.put((
            "manual_filter_complete",
            (
                gui.ManualInputFilterResult(
                    filtered_text="",
                    excluded_count=2,
                    remaining_count=0),
                ("pipeline",),
            ),
        ))
        app._set_generation_busy = MagicMock()
        app._set_status = MagicMock()
        app._start_pipeline_generation = MagicMock()
        app.root = MagicMock()

        with patch.object(gui.messagebox, "showinfo") as showinfo:
            app._poll_result()

        showinfo.assert_called_once()
        app._start_pipeline_generation.assert_not_called()
        app._set_generation_busy.assert_called_once_with(False)

    def test_manual_input_filter_rejects_inconsistent_counts(self):
        with self.assertRaisesRegex(ValueError, "remaining_count"):
            gui.normalise_manual_input_filter_response(
                {
                    "filtered_text": "one\ntwo",
                    "excluded_count": 1,
                    "remaining_count": 1,
                },
                2)
        with self.assertRaisesRegex(ValueError, "requested lines"):
            gui.normalise_manual_input_filter_response(
                {
                    "filtered_text": "one",
                    "excluded_count": 0,
                    "remaining_count": 1,
                },
                2)

    def test_file_preparation_languages_match_supported_tokenizers(self):
        self.assertEqual(
            gui.PREPARABLE_SOURCE_LANGUAGE_NAMES,
            (
                "Classical Chinese (Warring States)",
                "Classical Chinese (Ming)",
            ))

    def test_scheduling_estimate_immediately_invalidates_inflight_result(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.source_estimate_price = MagicMock()
        app.source_estimate_detail = MagicMock()
        app.source_estimate_generation = 4
        app.source_estimate_result = {"candidate_count": 500}
        app.source_paid_authorized = MagicMock()
        app.source_estimate_after_id = None
        app._update_source_generate_button_state = MagicMock()
        app.root = MagicMock()
        app.root.after.return_value = "replacement-estimate"

        app._schedule_source_estimate()

        self.assertEqual(app.source_estimate_generation, 5)
        self.assertIsNone(app.source_estimate_result)
        app.source_paid_authorized.set.assert_called_once_with(False)
        app.root.after.assert_called_once_with(
            250,
            app._refresh_source_estimate)
        self.assertEqual(
            app.source_estimate_after_id,
            "replacement-estimate")

    def test_source_estimate_formats_range_and_accounting(self):
        price, detail = gui.format_source_estimate({
            "estimated_cost_low_usd": 2.345,
            "estimated_cost_high_usd": 4.678,
            "candidate_count": 922,
            "request_count": 2,
            "input_tokens": 12345,
            "output_tokens": 67890,
            "largest_request_input_tokens": 7000,
            "largest_request_output_tokens": 34000,
            "request_limit_warning": "Reduce this request.",
        })

        self.assertEqual(price, "A$3.36–A$6.71")
        self.assertIn("922 new words", detail)
        self.assertIn("2 requests", detail)
        self.assertIn("12,345 estimated input tokens", detail)
        self.assertIn("67,890 estimated output tokens", detail)
        self.assertIn(
            "largest request ≈ 7,000 input / 34,000 output",
            detail)
        self.assertIn(
            "automatic transient retries are not included",
            detail)
        self.assertIn("Reduce this request", detail)

    def test_learned_filter_note_types_are_applied_to_the_selected_row(self):
        app = object.__new__(gui.AutoAnkiApp)
        row = {
            "note_var": MagicMock(
                get=MagicMock(return_value="Basic")),
            "field_var": MagicMock(),
            "note_box": MagicMock(),
            "field_box": MagicMock(),
        }
        app.source_exclusion_status = MagicMock()
        app._request_learned_filter_options = MagicMock()
        app._save_learned_filter_rows = MagicMock()

        app._apply_learned_filter_options(
            row,
            "note_types",
            {"note_types": ("Basic", "Cloze")})

        row["note_box"].configure.assert_called_once_with(
            values=("Basic", "Cloze"),
            state="readonly")
        app._request_learned_filter_options.assert_called_once_with(
            row,
            "note_type_details",
            "Basic")
        app.source_exclusion_status.set.assert_called_once_with(
            "Loaded 2 note types from the deck.")

    def test_learned_filter_fields_follow_the_selected_note_type(self):
        app = object.__new__(gui.AutoAnkiApp)
        row = {
            "field_var": MagicMock(
                get=MagicMock(return_value="Front")),
            "field_box": MagicMock(),
        }
        app.source_exclusion_status = MagicMock()
        app._save_learned_filter_rows = MagicMock()

        app._apply_learned_filter_options(
            row,
            "note_type_details",
            {
                "fields": ("Front", "Back"),
                "templates": ("Card 1", "Card 2"),
            })

        row["field_box"].configure.assert_called_once_with(
            values=("Back", "Front"),
            state="readonly")
        app.source_exclusion_status.set.assert_called_once_with(
            "Loaded 2 fields from the note type.")

    def test_learned_filter_deck_selection_uses_row_specific_async_contract(
            self):
        app = object.__new__(gui.AutoAnkiApp)
        row = {
            "deck_var": MagicMock(
                get=MagicMock(return_value="Learned French")),
            "note_var": MagicMock(),
            "field_var": MagicMock(),
            "note_box": MagicMock(),
            "field_box": MagicMock(),
        }
        app._request_learned_filter_options = MagicMock()
        app._save_learned_filter_rows = MagicMock()

        app._learned_filter_deck_selected(row)

        app._request_learned_filter_options.assert_called_once_with(
            row,
            "note_types",
            "Learned French")
        row["note_var"].set.assert_called_once_with("")
        row["field_var"].set.assert_called_once_with("")

    def test_learned_filter_note_selection_requests_fields_for_that_row(self):
        app = object.__new__(gui.AutoAnkiApp)
        row = {
            "note_var": MagicMock(
                get=MagicMock(return_value="Vocabulary")),
            "field_var": MagicMock(),
            "field_box": MagicMock(),
        }
        app._request_learned_filter_options = MagicMock()
        app._save_learned_filter_rows = MagicMock()

        app._learned_filter_note_selected(row)

        app._request_learned_filter_options.assert_called_once_with(
            row,
            "note_type_details",
            "Vocabulary")

    def test_source_request_includes_context_card_setup_and_safe_deck_policy(
            self):
        app = object.__new__(gui.AutoAnkiApp)
        option = gui.SourceUiOption(
            key="journey_to_the_west",
            name="Journey to the West",
            source_language_key="classical_chinese_ming")
        app._selected_source_option = MagicMock(return_value=option)
        app.source_chunk_size = MagicMock(
            get=MagicMock(return_value="200"))
        app.source_concurrency = MagicMock(
            get=MagicMock(return_value="6"))
        app.source_request_stagger_ms = MagicMock(
            get=MagicMock(return_value="100"))
        app.source_context_label = MagicMock(
            get=MagicMock(
                return_value=gui.SOURCE_CONTEXT_LABELS[
                    "sentence_neighbors"]))
        app.source_language_label = MagicMock(
            get=MagicMock(
                return_value="Classical Chinese (Ming)"))
        app.source_allow_web_search = MagicMock(
            get=MagicMock(return_value=True))
        pipeline = configured_pipeline("classical_chinese_ming")
        app.get_pipeline_configs = MagicMock(
            return_value=(pipeline,))
        app._selected_anki_exclusions = MagicMock(return_value=(
            {
                "deck": "Learned",
                "model": "Basic",
                "field": "Word",
            },
            {
                "deck": "Learned",
                "model": "Classical Chinese",
                "field": "Expression",
            },
        ))

        request = app._source_request()

        self.assertEqual(request["chunk_size"], 200)
        self.assertEqual(request["concurrency"], 6)
        self.assertEqual(request["request_stagger_ms"], 100)
        self.assertEqual(
            request["context_mode"],
            "sentence_neighbors")
        self.assertTrue(request["allow_web_search"])
        self.assertEqual(
            request["pipeline"].language_key,
            "classical_chinese_ming")
        self.assertEqual(
            request["anki_exclusions"],
            (
                {
                    "deck": "Learned",
                    "model": "Basic",
                    "field": "Word",
                },
                {
                    "deck": "Learned",
                    "model": "Classical Chinese",
                    "field": "Expression",
                },
            ))
        self.assertEqual(
            request["output_deck_name"],
            "Vocabulary from Journey to the West")
        self.assertEqual(
            request["source_name"],
            "Journey to the West")
        self.assertTrue(request["keep_imported_deck"])
        self.assertFalse(request["move_cards_after_import"])
        self.assertFalse(request["delete_imported_deck"])

    def test_zero_new_source_words_never_dispatch_paid_generation(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.source_paid_authorized = MagicMock(
            get=MagicMock(return_value=True))
        app._source_request = MagicMock(return_value={"source_key": "x"})
        app.source_estimate_result = {"candidate_count": 0}
        app._dispatch_source_action = MagicMock()

        with patch.object(gui.messagebox, "showinfo") as showinfo:
            app._start_source_generation()

        showinfo.assert_called_once()
        app._dispatch_source_action.assert_not_called()

    def test_failed_source_retry_requires_manual_paid_authorization(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.source_retry_callback = MagicMock()
        app._dispatch_source_action = MagicMock()
        records = ({"job_id": "chunk-1"},)

        with patch.object(
                gui.messagebox,
                "askyesno",
                return_value=False):
            app._confirm_and_retry_source_jobs(records)

        app._dispatch_source_action.assert_not_called()

        with patch.object(
                gui.messagebox,
                "askyesno",
                return_value=True):
            app._confirm_and_retry_source_jobs(records)

        app._dispatch_source_action.assert_called_once_with(
            "retry",
            app.source_retry_callback,
            {
                "job_ids": ("chunk-1",),
                "paid_confirmed": True,
            })

    def test_recoverable_finalize_row_is_retryable_in_the_gui(self):
        self.assertTrue(gui.AutoAnkiApp._source_job_is_retryable({
            "job_id": "saved-job::finalize",
            "status": "failed",
        }))
        self.assertFalse(gui.AutoAnkiApp._source_job_is_retryable({
            "job_id": "saved-job::finalize",
            "status": "pending",
        }))

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
                state="normal")
        editor.shared_field_containers[
            "english"].grid_remove.assert_called_once()
        for direction in pipeline_store.list_directions():
            editor.direction_target_deck_boxes[
                "english"][direction.key].configure.assert_called_with(
                    state="disabled")
            editor.per_card_field_containers[
                "english"][direction.key].grid.assert_called_once()
        editor._refresh_layout_geometry.assert_called_once_with(
            "english")

    def test_target_deck_mousewheel_scrolls_page_without_changing_deck(self):
        editor = object.__new__(gui.PipelineEditor)
        shared_box = MagicMock()
        first_card_box = MagicMock()
        second_card_box = MagicMock()
        editor.target_deck_boxes = {"english": shared_box}
        editor.direction_target_deck_boxes = {
            "english": {
                "context": first_card_box,
                "word_to_meaning": second_card_box,
            },
        }
        editor.app = MagicMock()

        editor.bind_target_deck_mousewheel()

        for box in (shared_box, first_card_box, second_card_box):
            self.assertEqual(
                tuple(call.args[0] for call in box.bind.call_args_list),
                ("<MouseWheel>", "<Button-4>", "<Button-5>"))
        callback = shared_box.bind.call_args_list[0].args[1]
        result = callback(SimpleNamespace(delta=-120, num=None))
        editor.app.pipeline_scroll_frame.canvas.yview_scroll \
            .assert_called_once_with(1, "units")
        self.assertEqual(result, "break")

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
