import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anki_integration
import templates


class AnkiConnectClientTests(unittest.TestCase):
    def test_invoke_sends_versioned_request_and_returns_result(self):
        response = io.BytesIO(json.dumps({
            "result": ["Default", "Modern English"],
            "error": None,
        }).encode("utf-8"))
        client = anki_integration.AnkiConnectClient(timeout=3)

        with patch.object(
                anki_integration.urllib.request,
                "urlopen",
                return_value=response) as urlopen:
            result = client.invoke("deckNames")

        self.assertEqual(result, ["Default", "Modern English"])
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload, {
            "action": "deckNames",
            "version": 6,
            "params": {},
        })
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 3)

    def test_invoke_raises_for_anki_connect_error(self):
        response = io.BytesIO(json.dumps({
            "result": None,
            "error": "sync: auth not configured",
        }).encode("utf-8"))
        client = anki_integration.AnkiConnectClient()

        with (
                patch.object(
                    anki_integration.urllib.request,
                    "urlopen",
                    return_value=response),
                self.assertRaisesRegex(
                    anki_integration.AnkiConnectResponseError,
                    "sync: auth not configured")):
            client.invoke("sync")

    def test_http_service_on_configured_port_is_identified(self):
        client = anki_integration.AnkiConnectClient(
            url="http://127.0.0.1:8765")
        error = anki_integration.urllib.error.HTTPError(
            client.url,
            501,
            "Unsupported method",
            {"Server": "InformerBrowser/1"},
            io.BytesIO(b"not AnkiConnect"))

        with (
                patch.object(
                    anki_integration.urllib.request,
                    "urlopen",
                    side_effect=error),
                self.assertRaisesRegex(
                    anki_integration.AnkiConnectResponseError,
                    "HTTP 501.*InformerBrowser.*Another service")):
            client.invoke("version")

    def test_non_json_service_on_configured_port_is_identified(self):
        client = anki_integration.AnkiConnectClient(
            url="http://127.0.0.1:8765")
        response = io.BytesIO(b"<html>not AnkiConnect</html>")

        with (
                patch.object(
                    anki_integration.urllib.request,
                    "urlopen",
                    return_value=response),
                self.assertRaisesRegex(
                    anki_integration.AnkiConnectResponseError,
                    "invalid JSON.*Another service")):
            client.invoke("version")

    def test_connection_settings_are_stored_with_owner_only_permissions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "anki_connection.json"
            saved = anki_integration.save_anki_connection_settings(
                "http://192.168.1.25:8765/",
                "  shared-secret  ",
                path=path)
            loaded = anki_integration.load_anki_connection_settings(path)

            self.assertEqual(
                saved,
                anki_integration.AnkiConnectionSettings(
                    url="http://192.168.1.25:8765",
                    api_key="shared-secret"))
            self.assertEqual(loaded, saved)
            if os.name == "posix":
                self.assertEqual(
                    stat.S_IMODE(path.stat().st_mode),
                    0o600)

    def test_saved_connection_is_used_by_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_directory = Path(temporary_directory) / "config"
            with patch.dict(
                    os.environ,
                    {
                        "AUTOANKI_CONFIG_DIR": str(config_directory),
                        "ANKICONNECT_API_KEY": "",
                        "AUTOANKI_ANKI_CONNECT_URL": "",
                    }):
                anki_integration.save_anki_connection_settings(
                    "http://anki-study.local:8765",
                    "saved-key")
                client = anki_integration.AnkiConnectClient()

            self.assertEqual(
                client.url,
                "http://anki-study.local:8765")
            self.assertEqual(client.api_key, "saved-key")
            self.assertFalse(client.is_local)

    def test_invalid_connection_url_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "http"):
            anki_integration.save_anki_connection_settings(
                "file:///tmp/anki")


class AnkiStartupTests(unittest.TestCase):
    def test_anki_window_ids_are_identified_by_window_class(self):
        result = MagicMock()
        result.stdout = (
            "0x001  0 Navigator.firefox host Firefox\n"
            "0x002  0 anki.anki host User 1 - Anki\n")

        with (
                patch.dict(os.environ, {"DISPLAY": ":0"}),
                patch.object(
                    anki_integration.shutil,
                    "which",
                    return_value="/usr/bin/wmctrl"),
                patch.object(
                    anki_integration.subprocess,
                    "run",
                    return_value=result)):
            window_ids = anki_integration.get_anki_window_ids()

        self.assertEqual(window_ids, {"0x002"})

    def test_only_new_anki_windows_are_minimized(self):
        with (
                patch.object(
                    anki_integration,
                    "get_anki_window_ids",
                    return_value={"0x-old", "0x-new"}),
                patch.object(
                    anki_integration,
                    "iconify_x11_window",
                    return_value=True) as iconify):
            minimized = anki_integration.minimize_new_anki_windows(
                {"0x-old"})

        self.assertTrue(minimized)
        iconify.assert_called_once_with("0x-new")

    def test_running_anki_is_not_launched_again(self):
        client = MagicMock()
        client.is_available.return_value = True
        launch = MagicMock()

        launched = anki_integration.ensure_anki_running(
            client,
            launch=launch)

        self.assertFalse(launched)
        launch.assert_not_called()

    def test_anki_is_launched_and_polled_until_available(self):
        client = MagicMock()
        client.is_available.side_effect = [False, False, True]
        launch = MagicMock()
        sleep = MagicMock()
        get_window_ids = MagicMock(return_value={"0x-existing"})
        minimize_new_windows = MagicMock(return_value=True)

        launched = anki_integration.ensure_anki_running(
            client,
            launch=launch,
            get_window_ids=get_window_ids,
            minimize_new_windows=minimize_new_windows,
            timeout=1,
            poll_interval=0,
            sleep=sleep)

        self.assertTrue(launched)
        launch.assert_called_once_with()
        get_window_ids.assert_called_once_with()
        minimize_new_windows.assert_called_once_with(
            {"0x-existing"})
        sleep.assert_called_once_with(0)

    def test_wrong_local_service_does_not_launch_or_wait(self):
        client = MagicMock()
        client.is_available.side_effect = (
            anki_integration.AnkiConnectResponseError(
                "Another service is using the port."))
        launch = MagicMock()

        with self.assertRaisesRegex(
                anki_integration.AnkiConnectResponseError,
                "Another service"):
            anki_integration.ensure_anki_running(
                client,
                launch=launch)

        launch.assert_not_called()
        client.is_available.assert_called_once_with(
            raise_response_errors=True)

    def test_remote_anki_is_never_launched_locally(self):
        client = anki_integration.AnkiConnectClient(
            url="http://192.168.1.25:8765",
            api_key="test-key")
        launch = MagicMock()
        with (
                patch.object(client, "is_available", return_value=False),
                self.assertRaisesRegex(
                    anki_integration.AnkiConnectUnavailableError,
                    "remote AnkiConnect")):
            anki_integration.ensure_anki_running(
                client,
                launch=launch)

        launch.assert_not_called()

    def test_collection_readiness_retries_while_syncing(self):
        client = MagicMock()
        client.invoke.side_effect = [
            anki_integration.AnkiConnectResponseError(
                "AnkiConnect error: collection is syncing"),
            ["Default", "Modern English"],
        ]
        sleep = MagicMock()
        times = iter([0, 0, 0])

        deck_names = anki_integration.wait_for_collection_ready(
            client,
            timeout=1,
            poll_interval=0,
            sleep=sleep,
            monotonic=lambda: next(times))

        self.assertEqual(deck_names, ["Default", "Modern English"])
        self.assertEqual(
            client.invoke.call_args_list,
            [call("deckNames"), call("deckNames")])
        sleep.assert_called_once_with(0)


class VocabularyExclusionTests(unittest.TestCase):
    def test_reads_exact_visible_terms_from_selected_card_field(self):
        client = MagicMock()
        client.invoke.side_effect = [
            [10, 11],
            [{"note": 100}, {"note": 101}],
            [
                {"fields": {"Word": {"value": "<b>道</b>"}}},
                {"fields": {"Word": {"value": "無&nbsp;為"}}},
            ],
        ]

        terms = anki_integration.read_existing_vocabulary(
            client,
            anki_integration.AnkiVocabularySource(
                deck_name='Retained "Chinese"',
                note_type_name="AutoAnki Classical Chinese",
                field_name="Word",
                card_template_name="Recognition"))

        self.assertEqual(terms, frozenset({"道", "無 為"}))
        self.assertEqual(client.invoke.call_args_list, [
            call(
                "findCards",
                query=(
                    'deck:"Retained \\"Chinese\\"" '
                    'note:"AutoAnki Classical Chinese" '
                    'card:"Recognition"')),
            call("cardsInfo", cards=[10, 11]),
            call("notesInfo", notes=[100, 101]),
        ])

    def test_lists_note_types_fields_and_templates_without_mutation(self):
        client = MagicMock()
        client.invoke.side_effect = [
            [1, 2],
            [
                {"modelName": "Second"},
                {"modelName": "First"},
            ],
            ["Word", "Meaning"],
            {
                "Recognition": {},
                "Production": {},
            },
        ]

        self.assertEqual(
            anki_integration.list_note_types_in_deck(
                client,
                "Vocabulary"),
            ("First", "Second"))
        self.assertEqual(
            anki_integration.get_note_type_fields(client, "First"),
            ("Word", "Meaning"))
        self.assertEqual(
            anki_integration.get_note_type_card_templates(
                client,
                "First"),
            ("Production", "Recognition"))


class ImportWorkflowTests(unittest.TestCase):
    def _create_package(self, directory):
        package_path = Path(directory) / "output.apkg"
        package_path.write_bytes(b"test package")
        return package_path

    def test_import_workflow_moves_and_deletes_without_manual_sync(self):
        client = MagicMock()
        client.invoke.side_effect = [
            True,
            [101, 102],
            None,
            [],
            [anki_integration.TARGET_DECK_NAME, templates.DECK_NAME],
            None,
        ]
        ensure_running = MagicMock()
        wait_until_ready = MagicMock(
            return_value=[
                "Default",
                anki_integration.TARGET_DECK_NAME])

        with tempfile.TemporaryDirectory() as temporary_directory:
            package_path = self._create_package(temporary_directory)
            result = anki_integration.import_generated_deck(
                package_path,
                client=client,
                ensure_running=ensure_running,
                wait_until_ready=wait_until_ready)

        self.assertEqual(
            result,
            anki_integration.AnkiImportResult(
                cards_moved=2,
                target_deck=anki_integration.TARGET_DECK_NAME,
                target_decks=(
                    anki_integration.TARGET_DECK_NAME,)))
        ensure_running.assert_called_once_with(client)
        wait_until_ready.assert_called_once_with(client)
        source_query = f'deck:"{templates.DECK_NAME}"'
        self.assertEqual(client.invoke.call_args_list, [
            call("importPackage", path=str(package_path.resolve())),
            call("findCards", query=source_query),
            call(
                "changeDeck",
                cards=[101, 102],
                deck=anki_integration.TARGET_DECK_NAME),
            call("findCards", query=source_query),
            call("deckNames"),
            call(
                "deleteDecks",
                decks=[templates.DECK_NAME],
                cardsToo=True),
        ])

    def test_standalone_import_does_not_move_or_delete_the_source_deck(self):
        client = MagicMock()
        client.invoke.return_value = True
        ensure_running = MagicMock()
        wait_until_ready = MagicMock(
            return_value=["Vocabulary from Source"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            package_path = self._create_package(temporary_directory)
            result = anki_integration.import_standalone_deck(
                package_path,
                client=client,
                ensure_running=ensure_running,
                wait_until_ready=wait_until_ready)

        self.assertTrue(result)
        ensure_running.assert_called_once_with(client)
        wait_until_ready.assert_called_once_with(client)
        self.assertEqual(
            client.invoke.call_args_list,
            [call(
                "importPackage",
                path=str(package_path.resolve()))])

    def test_missing_destination_deck_stops_before_import(self):
        client = MagicMock()
        wait_until_ready = MagicMock(return_value=["Default"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            package_path = self._create_package(temporary_directory)
            with self.assertRaisesRegex(
                    anki_integration.AnkiIntegrationError,
                    "does not exist"):
                anki_integration.import_generated_deck(
                    package_path,
                    client=client,
                    ensure_running=MagicMock(),
                    wait_until_ready=wait_until_ready)

        client.invoke.assert_not_called()

    def test_remote_package_is_uploaded_imported_and_removed(self):
        client = MagicMock()
        client.is_local = False
        client.invoke.side_effect = [
            "_autoanki_import_test.apkg",
            r"C:\Users\Study\AppData\Roaming\Anki2\User 1\collection.media",
            True,
            None,
        ]

        with (
                tempfile.TemporaryDirectory() as temporary_directory,
                patch.object(
                    anki_integration.uuid,
                    "uuid4",
                    return_value=MagicMock(hex="test"))):
            package_path = self._create_package(temporary_directory)
            imported = anki_integration.import_package(
                client,
                package_path)

        self.assertTrue(imported)
        calls = client.invoke.call_args_list
        self.assertEqual(calls[0].args, ("storeMediaFile",))
        self.assertEqual(
            calls[0].kwargs["filename"],
            "_autoanki_import_test.apkg")
        self.assertEqual(
            calls[0].kwargs["data"],
            "dGVzdCBwYWNrYWdl")
        self.assertEqual(calls[1], call("getMediaDirPath"))
        self.assertEqual(
            calls[2],
            call(
                "importPackage",
                path=(
                    "C:\\Users\\Study\\AppData\\Roaming\\Anki2\\User 1"
                    "\\collection.media/_autoanki_import_test.apkg")))
        self.assertEqual(
            calls[3],
            call(
                "deleteMediaFile",
                filename="_autoanki_import_test.apkg"))

    def test_remote_staging_file_is_removed_when_import_fails(self):
        client = MagicMock()
        client.is_local = False
        client.invoke.side_effect = [
            "_autoanki_import_test.apkg",
            "/anki/collection.media",
            anki_integration.AnkiConnectResponseError("import failed"),
            None,
        ]

        with (
                tempfile.TemporaryDirectory() as temporary_directory,
                patch.object(
                    anki_integration.uuid,
                    "uuid4",
                    return_value=MagicMock(hex="test"))):
            package_path = self._create_package(temporary_directory)
            with self.assertRaisesRegex(
                    anki_integration.AnkiConnectResponseError,
                    "import failed"):
                anki_integration.import_package(
                    client,
                    package_path)

        self.assertEqual(
            client.invoke.call_args_list[-1],
            call(
                "deleteMediaFile",
                filename="_autoanki_import_test.apkg"))

    def test_move_retries_until_temporary_deck_is_empty(self):
        client = MagicMock()
        client.invoke.side_effect = [
            [101],
            None,
            [101],
            None,
            [],
        ]
        sleep = MagicMock()

        cards_moved = anki_integration.move_all_cards(
            client,
            source_deck=templates.DECK_NAME,
            target_deck=anki_integration.TARGET_DECK_NAME,
            sleep=sleep)

        self.assertEqual(cards_moved, 1)
        self.assertEqual(
            [
                invocation.args[0]
                for invocation in client.invoke.call_args_list
            ],
            [
                "findCards",
                "changeDeck",
                "findCards",
                "changeDeck",
                "findCards",
            ])
        sleep.assert_called_once_with(
            anki_integration.MOVE_RETRY_DELAY_SECONDS)

    def test_import_routes_each_note_type_to_its_own_deck(self):
        client = MagicMock()
        client.invoke.side_effect = [
            True,
            [101],
            None,
            [],
            [202],
            None,
            [],
            [],
            ["AutoAnki Generated"],
            None,
        ]
        target_decks = (
            ("AutoAnki English vocabulary", "English::Context"),
            ("AutoAnki English word to meaning", "English::Recognition"),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            package_path = self._create_package(temporary_directory)
            result = anki_integration.import_generated_deck(
                package_path,
                target_decks=target_decks,
                source_deck="AutoAnki Generated",
                client=client,
                ensure_running=MagicMock(),
                wait_until_ready=MagicMock(return_value=[
                    "English::Context",
                    "English::Recognition",
                ]))

        self.assertEqual(result.cards_moved, 2)
        self.assertEqual(result.target_deck, "Multiple decks")
        self.assertEqual(
            result.target_decks,
            ("English::Context", "English::Recognition"))
        self.assertEqual(client.invoke.call_args_list, [
            call("importPackage", path=str(package_path.resolve())),
            call(
                "findCards",
                query=(
                    'deck:"AutoAnki Generated" '
                    'note:"AutoAnki English vocabulary"')),
            call(
                "changeDeck",
                cards=[101],
                deck="English::Context"),
            call(
                "findCards",
                query=(
                    'deck:"AutoAnki Generated" '
                    'note:"AutoAnki English vocabulary"')),
            call(
                "findCards",
                query=(
                    'deck:"AutoAnki Generated" '
                    'note:"AutoAnki English word to meaning"')),
            call(
                "changeDeck",
                cards=[202],
                deck="English::Recognition"),
            call(
                "findCards",
                query=(
                    'deck:"AutoAnki Generated" '
                    'note:"AutoAnki English word to meaning"')),
            call(
                "findCards",
                query='deck:"AutoAnki Generated"'),
            call("deckNames"),
            call(
                "deleteDecks",
                decks=["AutoAnki Generated"],
                cardsToo=True),
        ])

    def test_temporary_deck_is_not_deleted_if_cards_remain(self):
        client = MagicMock()
        client.invoke.side_effect = [
            True,
            [101],
            None,
            [101],
            None,
            [101],
            None,
            [101],
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            package_path = self._create_package(temporary_directory)
            with self.assertRaisesRegex(
                    anki_integration.AnkiIntegrationError,
                    "temporary deck was kept"):
                anki_integration.import_generated_deck(
                    package_path,
                    client=client,
                    ensure_running=MagicMock(),
                    wait_until_ready=MagicMock(
                        return_value=[
                            anki_integration.TARGET_DECK_NAME]))

        actions = [
            invocation.args[0]
            for invocation in client.invoke.call_args_list
        ]
        self.assertNotIn("deleteDecks", actions)


if __name__ == "__main__":
    unittest.main()
