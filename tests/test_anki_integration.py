import io
import json
import os
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
                target_deck=anki_integration.TARGET_DECK_NAME))
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
