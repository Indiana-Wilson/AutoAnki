import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import genanki


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import process_text
import templates
import gui


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
            result = process_text.get_api_key()

        self.assertEqual(result, "saved-key")

    def test_fetch_response_rejects_missing_api_key_before_openai_call(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            prompt_path = Path(temporary_directory) / "prompt"
            prompt_path.write_text("Prompt: ", encoding="utf-8")

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
                    prompt_path=prompt_path)

        openai_class.assert_not_called()

    def test_fetch_response_uses_client_and_writes_outputs(self):
        result = '{"cards": []}'
        client = MagicMock()
        response = client.responses.create.return_value
        response.status = "completed"
        response.output = []
        response.output_text = result

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            prompt_path = directory / "prompt"
            response_path = directory / "response.json"
            log_path = directory / "response_log"
            prompt_path.write_text("Define these words:\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                returned = process_text.fetch_response(
                    "astrolabe",
                    client=client,
                    prompt_path=prompt_path,
                    response_path=response_path,
                    response_log_path=log_path)

            self.assertEqual(returned, result)
            client.responses.create.assert_called_once_with(
                model="gpt-5.4-mini",
                input="Define these words:\nastrolabe",
                text={"format": process_text.build_response_format(
                    (templates.ENGLISH_VOCABULARY_CARD_TYPE,))})
            self.assertEqual(response_path.read_text(encoding="utf-8"), result)
            self.assertEqual(
                log_path.read_text(encoding="utf-8"),
                "\n<break>\n" + result)

    def test_fetch_response_constructs_openai_client_from_environment(self):
        fake_client = MagicMock()
        response = fake_client.responses.create.return_value
        response.status = "completed"
        response.output = []
        response.output_text = '{"cards": []}'

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            prompt_path = directory / "prompt"
            prompt_path.write_text("Prompt: ", encoding="utf-8")

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
                    prompt_path=prompt_path,
                    response_path=directory / "response.json",
                    response_log_path=directory / "response_log")

        openai_class.assert_called_once_with(
            api_key="test-key",
            max_retries=0)
        fake_client.responses.create.assert_called_once()

    def test_response_format_is_strict_and_uses_minimum_selected_fields(self):
        response_format = process_text.build_response_format((
            templates.ENGLISH_WORD_TO_MEANING_CARD_TYPE,
            templates.ENGLISH_MEANING_TO_WORD_CARD_TYPE,
        ))
        schema = response_format["schema"]
        card_schema = schema["properties"]["cards"]["items"]

        self.assertEqual(
            response_format["name"],
            "autoanki_english_simple_cards")
        self.assertTrue(response_format["strict"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            card_schema["required"],
            ["Word", "Meaning"])
        self.assertEqual(
            set(card_schema["properties"]),
            {"Word", "Meaning"})
        self.assertFalse(card_schema["additionalProperties"])

    def test_mixed_french_definitions_request_only_shared_required_fields(self):
        response_format = process_text.build_response_format((
            templates.FRENCH_WORD_TO_MEANING_CARD_TYPE,
            templates.FRENCH_WORD_TO_NATIVE_MEANING_CARD_TYPE,
        ))
        card_schema = (
            response_format["schema"]["properties"]["cards"]["items"])

        self.assertEqual(
            response_format["name"],
            "autoanki_french_simple_cards")
        self.assertEqual(
            card_schema["required"],
            ["French", "Meaning", "Native Meaning"])
        self.assertEqual(
            set(card_schema["properties"]),
            {"French", "Meaning", "Native Meaning"})

    def test_latin_native_detail_uses_latin_schema_name_and_fields(self):
        response_format = process_text.build_response_format((
            templates.LATIN_NATIVE_VOCABULARY_CARD_TYPE,
        ))
        card_schema = (
            response_format["schema"]["properties"]["cards"]["items"])

        self.assertEqual(
            response_format["name"],
            "autoanki_latin_detailed_cards")
        self.assertEqual(
            card_schema["required"],
            [
                "Latin",
                "Sentences",
                "Native Meaning",
                "Pronunciation",
            ])

    def test_fetch_response_rejects_refusal_without_writing_files(self):
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

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            prompt_path = directory / "prompt"
            response_path = directory / "response.json"
            log_path = directory / "response_log"
            prompt_path.write_text("Prompt: ", encoding="utf-8")

            with self.assertRaises(process_text.OpenAIRefusalError):
                process_text.fetch_response(
                    "word",
                    client=client,
                    prompt_path=prompt_path,
                    response_path=response_path,
                    response_log_path=log_path)

            self.assertFalse(response_path.exists())
            self.assertFalse(log_path.exists())

    def test_fetch_response_rejects_incomplete_response(self):
        client = MagicMock()
        response = client.responses.create.return_value
        response.status = "incomplete"
        response.output = []
        response.incomplete_details = {"reason": "max_output_tokens"}

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            prompt_path = directory / "prompt"
            prompt_path.write_text("Prompt: ", encoding="utf-8")

            with self.assertRaisesRegex(
                    process_text.OpenAIIncompleteResponseError,
                    "max_output_tokens"):
                process_text.fetch_response(
                    "word",
                    client=client,
                    prompt_path=prompt_path,
                    response_path=directory / "response.json",
                    response_log_path=directory / "response_log")

    def test_fetch_response_file_reads_words_and_returns_response(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            words_path = Path(temporary_directory) / "words"
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


class ProcessJsonTests(unittest.TestCase):
    def test_latin_response_creates_english_and_native_context_cards(self):
        text = json.dumps({"cards": [{
            "Latin": "sapientia",
            "Sentences": (
                "<strong>Sapientia</strong> ducem bonum facit.|"
                "Sine <strong>sapientia</strong> potentia periculosa est.|"
                "Philosophus <strong>sapientiam</strong> quaerit.|"
                "<strong>Sapientia</strong> aetate saepe crescit."),
            "Meaning": "Wisdom; sound knowledge and judgement.",
            "Native Meaning": (
                "Scientia rerum cum iudicio recto coniuncta."),
            "Pronunciation": "/sa.piˈen.ti.a/",
        }]})
        deck = MagicMock()
        package = MagicMock()
        card_types = (
            templates.LATIN_VOCABULARY_CARD_TYPE,
            templates.LATIN_NATIVE_VOCABULARY_CARD_TYPE,
        )

        with (
                patch.object(process_text.genanki, "Note") as note_class,
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            count = process_text.process_json_text(
                text,
                deck=deck,
                output_path="/tmp/latin.apkg",
                card_types=card_types,
                guid_seed="latin-pipeline")

        self.assertEqual(count, 2)
        self.assertEqual(
            [call.kwargs["model"] for call in note_class.call_args_list],
            [
                templates.latin_vocabulary_model,
                templates.latin_native_vocabulary_model,
            ])
        self.assertEqual(
            note_class.call_args_list[1].kwargs["fields"][2],
            "Scientia rerum cum iudicio recto coniuncta.")

    def test_french_response_creates_all_definition_directions(self):
        text = json.dumps({"cards": [{
            "French": "épanouir",
            "Sentences": (
                "La fleur va <strong>épanouir</strong> ses pétales.|"
                "Ce travail l'aide à <strong>épanouir</strong> son talent.|"
                "Le soleil fait <strong>épanouir</strong> le jardin.|"
                "Elle veut <strong>épanouir</strong> sa créativité."),
            "Meaning": "To cause to flourish or develop fully.",
            "Native Meaning": (
                "Faire se développer pleinement ou devenir florissant."),
            "Pronunciation": "/e.pa.nwiʁ/",
        }]})
        deck = MagicMock()
        package = MagicMock()
        card_types = (
            templates.FRENCH_VOCABULARY_CARD_TYPE,
            templates.FRENCH_WORD_TO_MEANING_CARD_TYPE,
            templates.FRENCH_MEANING_TO_WORD_CARD_TYPE,
            templates.FRENCH_NATIVE_VOCABULARY_CARD_TYPE,
            templates.FRENCH_WORD_TO_NATIVE_MEANING_CARD_TYPE,
            templates.FRENCH_NATIVE_MEANING_TO_WORD_CARD_TYPE,
        )

        with (
                patch.object(process_text.genanki, "Note") as note_class,
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            count = process_text.process_json_text(
                text,
                deck=deck,
                output_path="/tmp/french.apkg",
                card_types=card_types,
                guid_seed="french-pipeline")

        self.assertEqual(count, 6)
        self.assertEqual(note_class.call_count, 6)
        self.assertEqual(
            [call.kwargs["model"] for call in note_class.call_args_list],
            [card_type.model for card_type in card_types])
        self.assertEqual(
            note_class.call_args_list[4].kwargs["fields"],
            [
                "épanouir",
                "Faire se développer pleinement ou devenir florissant.",
            ])

    def test_japanese_native_definition_creates_expected_note(self):
        text = json.dumps({"cards": [{
            "Japanese": "勉強",
            "Native Meaning": "知識や技能を身につけるために学ぶこと。",
        }]})
        deck = MagicMock()
        package = MagicMock()

        with (
                patch.object(process_text.genanki, "Note") as note_class,
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            process_text.process_json_text(
                text,
                deck=deck,
                output_path="/tmp/japanese.apkg",
                card_type=(
                    templates
                    .JAPANESE_NATIVE_MEANING_TO_WORD_CARD_TYPE),
                guid_seed="japanese-pipeline")

        note_class.assert_called_once_with(
            model=templates.japanese_native_meaning_to_word_model,
            fields=[
                "勉強",
                "知識や技能を身につけるために学ぶこと。",
            ],
            guid=process_text.genanki.guid_for(
                "autoanki",
                "japanese-pipeline",
                "japanese_native_meaning_to_word",
                "勉強",
                "知識や技能を身につけるために学ぶこと。"))

    def test_one_detailed_response_creates_multiple_selected_card_types(self):
        text = json.dumps([{
            "Word": "astrolabe",
            "Sentences": "First|Second|Third|Fourth",
            "Meaning": "An astronomical instrument.",
            "Pronunciation": "/ˈæstrəleɪb/",
        }])
        deck = MagicMock()
        package = MagicMock()

        with (
                patch.object(process_text.genanki, "Note") as note_class,
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            count = process_text.process_json_text(
                text,
                deck=deck,
                output_path="/tmp/multiple.apkg",
                card_types=(
                    templates.ENGLISH_VOCABULARY_CARD_TYPE,
                    templates.ENGLISH_WORD_TO_MEANING_CARD_TYPE,
                    templates.ENGLISH_MEANING_TO_WORD_CARD_TYPE),
                guid_seed="pipeline-one")

        self.assertEqual(count, 3)
        self.assertEqual(note_class.call_count, 3)
        self.assertEqual(
            [
                call.kwargs["model"]
                for call in note_class.call_args_list
            ],
            [
                templates.my_model,
                templates.english_word_to_meaning_model,
                templates.english_meaning_to_word_model,
            ])
        self.assertEqual(
            note_class.call_args_list[1].kwargs["fields"],
            ["astrolabe", "An astronomical instrument."])
        self.assertEqual(deck.add_note.call_count, 3)
        package.write_to_file.assert_called_once_with(
            Path("/tmp/multiple.apkg"))

    def test_process_json_text_creates_classical_chinese_note(self):
        text = json.dumps([{
            "Classical Chinese": "學而時習之",
            "Sentences": (
                "<strong>學而時習之</strong>，不亦說乎？|"
                "孔子曰：<strong>學而時習之</strong>。|"
                "君子以<strong>學而時習之</strong>為樂。|"
                "弟子當<strong>學而時習之</strong>。"),
            "Meaning": "To study and regularly practise what one has learned.",
            "Pronunciation": "xué ér shí xí zhī",
        }])
        deck = MagicMock()
        package = MagicMock()

        with (
                patch.object(process_text.genanki, "Note") as note_class,
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            process_text.process_json_text(
                text,
                deck=deck,
                output_path="/tmp/classical-chinese.apkg",
                card_type=templates.CLASSICAL_CHINESE_CARD_TYPE,
                guid_seed="classical-pipeline")

        note_class.assert_called_once_with(
            model=templates.classical_chinese_model,
            fields=[
                "學而時習之",
                (
                    "<strong>學而時習之</strong>，不亦說乎？|"
                    "孔子曰：<strong>學而時習之</strong>。|"
                    "君子以<strong>學而時習之</strong>為樂。|"
                    "弟子當<strong>學而時習之</strong>。"),
                "To study and regularly practise what one has learned.",
                "xué ér shí xí zhī",
            ],
            guid=process_text.genanki.guid_for(
                "autoanki",
                "classical-pipeline",
                "classical_chinese_vocabulary",
                "學而時習之",
                "To study and regularly practise what one has learned."))
        deck.add_note.assert_called_once_with(note_class.return_value)

    def test_process_json_text_creates_note_with_expected_fields(self):
        text = json.dumps([{
            "Word": "astrolabe",
            "Sentences": "First|Second|Third|Fourth",
            "Meaning": "An astronomical instrument.",
            "Pronunciation": "/ˈæstrəleɪb/",
        }])
        deck = MagicMock()
        package = MagicMock()

        with (
                patch.object(process_text.genanki, "Note") as note_class,
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            process_text.process_json_text(
                text,
                deck=deck,
                output_path="/tmp/test-output.apkg")

        note_class.assert_called_once_with(
            model=templates.my_model,
            fields=[
                "astrolabe",
                "First|Second|Third|Fourth",
                "An astronomical instrument.",
                "/ˈæstrəleɪb/"])
        deck.add_note.assert_called_once_with(note_class.return_value)
        package.write_to_file.assert_called_once_with(
            Path("/tmp/test-output.apkg"))

    def test_process_json_text_rejects_invalid_json_before_export(self):
        deck = MagicMock()

        with (
                patch.object(process_text.genanki, "Package") as package_class,
                self.assertRaises(json.JSONDecodeError)):
            process_text.process_json_text("not JSON", deck=deck)

        deck.add_note.assert_not_called()
        package_class.assert_not_called()

    def test_process_json_text_rejects_missing_fields(self):
        text = json.dumps([{"Word": "incomplete"}])
        deck = MagicMock()

        with self.assertRaises(process_text.GeneratedCardValidationError):
            process_text.process_json_text(text, deck=deck)

        deck.add_note.assert_not_called()

    def test_process_json_text_accepts_structured_outputs_root_object(self):
        text = json.dumps({"cards": [{
            "Word": "astrolabe",
            "Meaning": "An astronomical instrument.",
        }]})
        deck = MagicMock()
        package = MagicMock()

        with (
                patch.object(process_text.genanki, "Note"),
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            count = process_text.process_json_text(
                text,
                deck=deck,
                output_path="/tmp/simple.apkg",
                card_type=templates.ENGLISH_WORD_TO_MEANING_CARD_TYPE)

        self.assertEqual(count, 1)
        deck.add_note.assert_called_once()

    def test_process_json_text_rejects_wrong_sentence_count_before_export(self):
        text = json.dumps({"cards": [{
            "Word": "astrolabe",
            "Sentences": "One|Two|Three",
            "Meaning": "An astronomical instrument.",
            "Pronunciation": "/ˈæstrəleɪb/",
        }]})
        deck = MagicMock()

        with (
                patch.object(process_text.genanki, "Package") as package_class,
                self.assertRaisesRegex(
                    process_text.GeneratedCardValidationError,
                    "exactly four")):
            process_text.process_json_text(text, deck=deck)

        deck.add_note.assert_not_called()
        package_class.assert_not_called()

    def test_pipeline_guid_is_stable_and_scoped_to_pipeline(self):
        note_data = {
            "Word": "astrolabe",
            "Sentences": "One|Two|Three|Four",
            "Meaning": "An astronomical instrument.",
            "Pronunciation": "/ˈæstrəleɪb/",
        }
        text = json.dumps([note_data])
        deck = MagicMock()
        package = MagicMock()

        with (
                patch.object(process_text.genanki, "Note") as note_class,
                patch.object(
                    process_text.genanki,
                    "Package",
                    return_value=package)):
            process_text.process_json_text(
                text,
                deck=deck,
                output_path="/tmp/test-output.apkg",
                guid_seed="pipeline-one")

        expected_guid = process_text.genanki.guid_for(
            "autoanki",
            "pipeline-one",
            templates.DEFAULT_CARD_TYPE_KEY,
            note_data["Word"],
            note_data["Meaning"])
        self.assertEqual(
            note_class.call_args.kwargs["guid"],
            expected_guid)
        other_pipeline_guid = process_text.genanki.guid_for(
            "autoanki",
            "pipeline-two",
            templates.DEFAULT_CARD_TYPE_KEY,
            note_data["Word"],
            note_data["Meaning"])
        self.assertNotEqual(
            expected_guid,
            other_pipeline_guid)

    def test_process_json_text_creates_a_real_anki_package(self):
        text = json.dumps([{
            "Word": "vermilion",
            "Sentences": "One|Two|Three|Four",
            "Meaning": "A brilliant red pigment or colour.",
            "Pronunciation": "/vəˈmɪljən/",
        }])
        deck = genanki.Deck(2059400111, "Test Generated Words")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "output.apkg"
            process_text.process_json_text(
                text,
                deck=deck,
                output_path=output_path)

            self.assertTrue(output_path.is_file())
            with zipfile.ZipFile(output_path) as package:
                self.assertIn("collection.anki2", package.namelist())
                self.assertIn("media", package.namelist())

    def test_process_json_file_reads_and_processes_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            response_path = Path(temporary_directory) / "response.json"
            response_path.write_text("[]", encoding="utf-8")

            with patch.object(
                    process_text,
                    "process_json_text") as process_json_text:
                process_text.process_json_file(
                    response_path,
                    output_path="deck.apkg")

        process_json_text.assert_called_once_with(
            "[]",
            output_path="deck.apkg")


class GenerateDeckTests(unittest.TestCase):
    def test_generate_deck_uses_a_fresh_deck_and_returns_output_path(self):
        fresh_deck = MagicMock()

        with (
                patch.object(
                    process_text,
                    "fetch_response",
                    return_value="[]") as fetch_response,
                patch.object(
                    process_text.templates,
                    "create_deck",
                    return_value=fresh_deck) as create_deck,
                patch.object(
                    process_text,
                    "process_json_text") as process_json_text):
            result = process_text.generate_deck(
                "astrolabe",
                client="fake-client",
                output_path="custom.apkg")

        self.assertEqual(result, Path("custom.apkg"))
        fetch_response.assert_called_once_with(
            "astrolabe",
            client="fake-client",
            prompt_path=None,
            response_path=process_text.RESPONSE_PATH,
            response_log_path=process_text.RESPONSE_LOG_PATH,
            response_format=process_text.build_response_format(
                (templates.ENGLISH_VOCABULARY_CARD_TYPE,)))
        create_deck.assert_called_once_with(
            templates.DECK_ID,
            templates.DECK_NAME)
        process_json_text.assert_called_once_with(
            "[]",
            deck=fresh_deck,
            output_path=Path("custom.apkg"),
            card_types=(templates.ENGLISH_VOCABULARY_CARD_TYPE,),
            guid_seed=None,
            legacy_guid_card_type_key=None)

    def test_create_deck_returns_a_new_empty_deck_each_time(self):
        first = templates.create_deck()
        second = templates.create_deck()

        self.assertIsNot(first, second)
        self.assertEqual(first.deck_id, templates.DECK_ID)
        self.assertEqual(second.deck_id, templates.DECK_ID)
        self.assertEqual(first.notes, [])
        self.assertEqual(second.notes, [])


class EntrypointTests(unittest.TestCase):
    def test_main_reads_words_then_generates_deck(self):
        with (
                patch.object(
                    type(process_text.WORDS_PATH),
                    "read_text",
                    return_value="one\ntwo") as read_text,
                patch.object(
                    process_text,
                    "generate_deck") as generate_deck):
            process_text.main()

        read_text.assert_called_once_with(encoding="utf-8")
        generate_deck.assert_called_once_with("one\ntwo")


class GuiWorkerTests(unittest.TestCase):
    def test_wheel_events_are_normalized_across_tk_platforms(self):
        self.assertEqual(
            gui.wheel_scroll_amount(MagicMock(num=4, state=0)),
            -1)
        self.assertEqual(
            gui.wheel_scroll_amount(MagicMock(num=5, state=0)),
            1)
        self.assertEqual(
            gui.wheel_scroll_amount(
                MagicMock(num=None, delta=120)),
            -1)

    def test_text_selection_is_cleared_when_focus_moves_away(self):
        widget = MagicMock()
        event = MagicMock(widget=widget)

        gui.AutoAnkiApp._clear_text_selection(event)

        widget.tag_remove.assert_called_once_with(
            gui.tk.SEL,
            "1.0",
            gui.tk.END)

    def test_entry_selection_is_cleared_when_focus_moves_away(self):
        widget = MagicMock()
        event = MagicMock(widget=widget)

        gui.AutoAnkiApp._clear_entry_selection(event)

        widget.selection_clear.assert_called_once_with()

    def test_background_click_clears_focus_but_text_click_does_not(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        background = MagicMock()
        background.winfo_class.return_value = "TFrame"
        text = MagicMock()
        text.winfo_class.return_value = "Text"

        app._clear_focus_on_background_click(
            MagicMock(widget=background))
        app._clear_focus_on_background_click(
            MagicMock(widget=text))

        app.root.focus_set.assert_called_once_with()

    def test_busy_state_updates_action_and_progress_feedback(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.generate_button = MagicMock()
        app.api_key_button = MagicMock()
        app.generation_progress = MagicMock()

        app._set_generation_busy(True)

        self.assertTrue(app.generation_in_progress)
        app.generate_button.configure.assert_called_once_with(
            state=gui.tk.DISABLED,
            text="Generating and importing…")
        app.api_key_button.configure.assert_called_once_with(
            state=gui.tk.DISABLED)
        app.generation_progress.start.assert_called_once_with(12)

    def test_close_waits_for_active_generation(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.generation_in_progress = True
        app.save_pipeline_rows = MagicMock()

        with patch.object(gui.messagebox, "showinfo") as showinfo:
            app.close()

        showinfo.assert_called_once()
        app.save_pipeline_rows.assert_not_called()
        app.root.destroy.assert_not_called()

    def test_prompt_editor_only_unlocks_after_checkbox_is_enabled(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.prompt_editing_enabled = MagicMock()
        app.prompt_editing_enabled.get.return_value = True
        app.prompt_text = MagicMock()
        app.prompt_dirty = False
        app.prompt_status = MagicMock()
        app.save_prompt_button = MagicMock()

        app._toggle_prompt_editing()

        app.prompt_text.configure.assert_called_once_with(
            state=gui.tk.NORMAL,
            foreground=app.TEXT_PRIMARY)
        app.prompt_text.focus_set.assert_called_once_with()
        app.save_prompt_button.configure.assert_called_once_with(
            state=gui.tk.DISABLED)

    def test_prompt_selector_uses_concrete_combobox_index(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.current_prompt_key = "english_vocab"
        app.prompt_options = (
            gui.pipeline_store.PromptOption(
                key="classical_chinese",
                name="Classical Chinese",
                path=Path("/prompts/classical_chinese")),
            gui.pipeline_store.PromptOption(
                key="english_vocab",
                name="English Vocab",
                path=Path("/prompts/english_vocab")),
        )
        app.prompt_selector = MagicMock()
        app.prompt_selector_value = MagicMock()

        app._refresh_prompt_selector_display()

        app.prompt_selector.current.assert_called_once_with(1)
        app.prompt_selector_value.set.assert_not_called()

    def test_disabling_prompt_editing_can_save_before_locking(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.prompt_editing_enabled = MagicMock()
        app.prompt_editing_enabled.get.return_value = False
        app.prompt_dirty = True
        app._confirm_unsaved_prompt = MagicMock(
            return_value="save")
        app.save_current_prompt = MagicMock(return_value=True)
        app.prompt_text = MagicMock()
        app._update_prompt_editor_state = MagicMock()

        app._toggle_prompt_editing()

        app.save_current_prompt.assert_called_once_with(
            show_confirmation=False,
            require_editing=False)
        app.prompt_text.configure.assert_called_once_with(
            state=gui.tk.DISABLED,
            foreground=app.TEXT_SECONDARY)

    def test_closing_with_unsaved_prompt_can_be_cancelled(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.generation_in_progress = False
        app.prompt_dirty = True
        app._confirm_unsaved_prompt = MagicMock(
            return_value="cancel")
        app.save_pipeline_rows = MagicMock()

        app.close()

        app._confirm_unsaved_prompt.assert_called_once_with(
            "closing AutoAnki")
        app.save_pipeline_rows.assert_not_called()
        app.root.destroy.assert_not_called()

    def test_close_flushes_settings_before_destroying_window(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.generation_in_progress = False
        app.pipeline_save_after_id = "pending-save"
        app.pipeline_notice_after_id = "pending-notice"
        app.save_pipeline_rows = MagicMock(return_value=("pipeline",))

        app.close()

        self.assertEqual(
            app.root.after_cancel.call_args_list,
            [
                call("pending-save"),
                call("pending-notice"),
            ])
        app.save_pipeline_rows.assert_called_once_with(
            show_errors=True)
        app.root.destroy.assert_called_once_with()

    def test_automatic_deck_refresh_does_not_launch_anki(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.deck_result_queue = gui.queue.Queue()
        client = MagicMock()
        client.is_available.return_value = False

        with patch.object(
                gui.anki_integration,
                "AnkiConnectClient",
                return_value=client):
            app._load_anki_decks_in_background()

        self.assertEqual(
            app.deck_result_queue.get_nowait(),
            ("unavailable", None))
        client.invoke.assert_not_called()

    def test_automatic_deck_refresh_updates_rows_and_schedules_next_poll(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.deck_result_queue = gui.queue.Queue()
        app.deck_result_queue.put(
            ("success", ("Parent", "Parent::Child")))
        app.deck_options = ("Old deck",)
        app.deck_refresh_in_progress = True
        app.pipeline_status = MagicMock()
        row = MagicMock()
        app.pipeline_rows = [row]

        with patch.object(
                gui.pipeline_store,
                "save_anki_deck_cache") as save_cache:
            app._poll_deck_result()

        self.assertFalse(app.deck_refresh_in_progress)
        self.assertEqual(
            app.deck_options,
            ("Parent", "Parent::Child"))
        row.update_deck_options.assert_called_once_with(
            ("Parent", "Parent::Child"))
        save_cache.assert_called_once_with(
            ("Parent", "Parent::Child"))
        app.root.after.assert_called_once_with(
            app.DECK_REFRESH_INTERVAL_MS,
            app.refresh_anki_decks)

    def test_separate_deck_mode_does_not_remove_or_regrid_card_rows(self):
        editor = object.__new__(gui.PipelineEditor)
        separate_mode = MagicMock()
        separate_mode.get.return_value = True
        selected = MagicMock()
        selected.get.return_value = True
        container = MagicMock()
        deck_box = MagicMock()
        shared_hint = MagicMock()
        editor.separate_target_deck_variables = {
            "japanese": separate_mode,
        }
        editor.shared_deck_labels = {
            "japanese": MagicMock(),
        }
        editor.target_deck_boxes = {
            "japanese": MagicMock(),
        }
        editor.card_target_deck_containers = {
            "japanese": {"japanese_vocabulary": container},
        }
        editor.card_target_deck_shared_hints = {
            "japanese": {"japanese_vocabulary": shared_hint},
        }
        editor.card_output_variables = {
            "japanese": {"japanese_vocabulary": selected},
        }
        editor.card_target_deck_boxes = {
            "japanese": {"japanese_vocabulary": deck_box},
        }

        editor._update_deck_mode_visibility("japanese")

        container.grid.assert_not_called()
        container.grid_remove.assert_not_called()
        deck_box.grid.assert_not_called()
        deck_box.grid_remove.assert_not_called()
        deck_box.configure.assert_called_once_with(state="normal")
        shared_hint.place_forget.assert_called_once_with()

    def test_pipeline_edits_are_debounced_and_saved_automatically(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.pipeline_save_after_id = "previous-save"

        app.schedule_pipeline_save()

        app.root.after_cancel.assert_called_once_with("previous-save")
        app.root.after.assert_called_once_with(
            400,
            app._autosave_pipeline_rows)
        self.assertIs(
            app.pipeline_save_after_id,
            app.root.after.return_value)

    def test_pipeline_autosave_is_silent_while_a_value_is_incomplete(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.pipeline_save_after_id = "pending-save"
        app.save_pipeline_rows = MagicMock()

        app._autosave_pipeline_rows()

        self.assertIsNone(app.pipeline_save_after_id)
        app.save_pipeline_rows.assert_called_once_with(show_errors=False)

    def test_ensure_api_key_prompts_when_no_key_is_configured(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.configure_api_key = MagicMock(return_value=True)

        with patch.object(process_text, "get_api_key", return_value=None):
            result = app._ensure_api_key()

        self.assertTrue(result)
        app.configure_api_key.assert_called_once_with(
            show_confirmation=False)

    def test_configure_api_key_saves_masked_dialog_value(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.status = MagicMock()

        with (
                patch.object(
                    gui.simpledialog,
                    "askstring",
                    return_value="test-api-key") as askstring,
                patch.object(
                    gui.credential_store,
                    "save_api_key") as save_api_key,
                patch.object(gui.messagebox, "showinfo")):
            result = app.configure_api_key()

        self.assertTrue(result)
        self.assertEqual(askstring.call_args.kwargs["show"], "*")
        save_api_key.assert_called_once_with("test-api-key")
        app.status.set.assert_called_once_with(
            "API key saved on this device.")

    def test_background_worker_queues_success(self):
        app = object.__new__(gui.AutoAnkiApp)
        pipeline = gui.pipeline_store.default_pipeline()
        import_result = gui.anki_integration.AnkiImportResult(
            cards_moved=3,
            target_deck=gui.anki_integration.TARGET_DECK_NAME)
        summary = gui.pipeline_runner.PipelineRunSummary(
            successes=(
                gui.pipeline_runner.PipelineSuccess(
                    pipeline=pipeline,
                    output_path=Path("/tmp/deck.apkg"),
                    import_result=import_result),),
            failures=())
        progress = gui.pipeline_runner.PipelineProgress(
            pipeline=pipeline,
            index=1,
            total=1,
            stage="generating")

        def execute(words, pipelines, progress_callback):
            self.assertEqual(words, "astrolabe")
            self.assertEqual(pipelines, (pipeline,))
            progress_callback(progress)
            return summary

        app.pipeline_executor = MagicMock(side_effect=execute)
        app.result_queue = gui.queue.Queue()

        app._generate_in_background(
            "astrolabe",
            (pipeline,))

        self.assertEqual(
            app.result_queue.get_nowait(),
            ("pipeline_progress", progress))
        self.assertEqual(
            app.result_queue.get_nowait(),
            ("run_complete", summary))

    def test_background_worker_queues_error(self):
        app = object.__new__(gui.AutoAnkiApp)
        error = RuntimeError("generation failed")
        app.pipeline_executor = MagicMock(side_effect=error)
        app.result_queue = gui.queue.Queue()

        app._generate_in_background(
            "astrolabe",
            (gui.pipeline_store.default_pipeline(),))

        outcome, queued_error = app.result_queue.get_nowait()
        self.assertEqual(outcome, "run_error")
        self.assertIs(queued_error, error)

    def test_run_summary_preserves_package_when_import_fails(self):
        pipeline = gui.pipeline_store.default_pipeline()
        error = RuntimeError("Anki unavailable")
        failure = gui.pipeline_runner.PipelineFailure(
            pipeline=pipeline,
            stage="import",
            error=error,
            output_path=Path("/tmp/deck.apkg"))
        self.assertEqual(
            failure.output_path,
            Path("/tmp/deck.apkg"))
        self.assertIs(failure.error, error)

    def test_poll_result_displays_import_failure_dialog(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.generate_button = MagicMock()
        app.status = MagicMock()
        app.output = MagicMock()
        app.result_queue = gui.queue.Queue()
        pipeline = gui.pipeline_store.default_pipeline()
        error = RuntimeError("Anki unavailable")
        summary = gui.pipeline_runner.PipelineRunSummary(
            successes=(),
            failures=(
                gui.pipeline_runner.PipelineFailure(
                    pipeline=pipeline,
                    stage="import",
                    error=error,
                    output_path=Path("/tmp/deck.apkg")),))
        app.result_queue.put((
            "run_complete",
            summary))

        with patch.object(
                gui.messagebox,
                "showerror") as showerror:
            app._poll_result()

        app.status.set.assert_called_once_with(
            "0 pipelines succeeded; 1 failed.")
        showerror.assert_called_once()
        self.assertEqual(
            showerror.call_args.args[0],
            "Some pipelines failed")
        self.assertIn(
            "Anki unavailable",
            showerror.call_args.args[1])

    def test_generation_failure_does_not_retry_without_authorization(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.generate_button = MagicMock()
        app.status = MagicMock()
        app.output = MagicMock()
        app.start_generation = MagicMock()
        app.result_queue = gui.queue.Queue()
        pipeline = gui.pipeline_store.default_pipeline()
        summary = gui.pipeline_runner.PipelineRunSummary(
            successes=(),
            failures=(
                gui.pipeline_runner.PipelineFailure(
                    pipeline=pipeline,
                    stage="generation",
                    error=RuntimeError("Invalid generated JSON")),))
        app.result_queue.put(("run_complete", summary))

        with patch.object(
                gui.messagebox,
                "askretrycancel",
                return_value=False) as askretrycancel:
            app._poll_result()

        askretrycancel.assert_called_once()
        self.assertIn(
            "paid OpenAI API request",
            askretrycancel.call_args.args[1])
        app.start_generation.assert_not_called()

    def test_generation_retries_only_after_manual_authorization(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.root = MagicMock()
        app.generate_button = MagicMock()
        app.status = MagicMock()
        app.output = MagicMock()
        app.start_generation = MagicMock()
        app.result_queue = gui.queue.Queue()
        pipeline = gui.pipeline_store.default_pipeline()
        summary = gui.pipeline_runner.PipelineRunSummary(
            successes=(),
            failures=(
                gui.pipeline_runner.PipelineFailure(
                    pipeline=pipeline,
                    stage="generation",
                    error=RuntimeError("Invalid generated JSON")),))
        app.result_queue.put(("run_complete", summary))

        with patch.object(
                gui.messagebox,
                "askretrycancel",
                return_value=True):
            app._poll_result()

        app.start_generation.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
