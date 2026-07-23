import ctypes
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import templates


ANKI_CONNECT_URL = "http://127.0.0.1:8765"
ANKI_CONNECT_VERSION = 6
ANKI_CONNECT_ADDON_CODE = "2055492159"
TARGET_DECK_NAME = "Retained Information::English::Modern English"
STARTUP_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 0.25
COLLECTION_READY_TIMEOUT_SECONDS = 300
MOVE_RETRY_COUNT = 3
MOVE_RETRY_DELAY_SECONDS = 0.2


class AnkiIntegrationError(RuntimeError):
    pass


class AnkiConnectUnavailableError(AnkiIntegrationError):
    pass


class AnkiConnectResponseError(AnkiIntegrationError):
    pass


@dataclass(frozen=True)
class AnkiImportResult:
    cards_moved: int
    target_deck: str


class AnkiConnectClient:
    def __init__(self, url=ANKI_CONNECT_URL, api_key=None, timeout=300):
        self.url = url
        self.api_key = api_key or os.environ.get("ANKICONNECT_API_KEY")
        self.timeout = timeout

    def invoke(self, action, **params):
        payload = {
            "action": action,
            "version": ANKI_CONNECT_VERSION,
            "params": params,
        }
        if self.api_key:
            payload["key"] = self.api_key

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(
                    request,
                    timeout=self.timeout) as response:
                result = json.load(response)
        except (OSError, urllib.error.URLError) as error:
            raise AnkiConnectUnavailableError(
                "Could not connect to AnkiConnect.") from error

        if not isinstance(result, dict):
            raise AnkiConnectResponseError(
                "AnkiConnect returned an invalid response.")
        if "error" not in result or "result" not in result:
            raise AnkiConnectResponseError(
                "AnkiConnect returned an incomplete response.")
        if result["error"] is not None:
            raise AnkiConnectResponseError(
                f"AnkiConnect error: {result['error']}")
        return result["result"]

    def is_available(self):
        try:
            return self.invoke("version") >= ANKI_CONNECT_VERSION
        except AnkiIntegrationError:
            return False


def get_anki_command():
    override = os.environ.get("AUTOANKI_ANKI_COMMAND")
    if override:
        command = shlex.split(override)
        if command:
            return command

    installed_command = shutil.which("anki")
    if installed_command:
        return [installed_command]

    if sys.platform == "darwin":
        return ["open", "-a", "Anki"]

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        program_files = os.environ.get("PROGRAMFILES")
        candidates = []
        if local_app_data:
            candidates.append(
                Path(local_app_data) / "Programs" / "Anki" / "anki.exe")
        if program_files:
            candidates.append(Path(program_files) / "Anki" / "anki.exe")
        for candidate in candidates:
            if candidate.is_file():
                return [str(candidate)]

    raise AnkiIntegrationError(
        "AutoAnki could not find the Anki application. Set "
        "AUTOANKI_ANKI_COMMAND to the command that starts Anki.")


def launch_anki(command=None):
    command = command or get_anki_command()
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name == "posix")


def get_anki_window_ids():
    wmctrl = shutil.which("wmctrl")
    if not wmctrl or not os.environ.get("DISPLAY"):
        return set()

    try:
        result = subprocess.run(
            [wmctrl, "-lx"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=3,
            check=False)
    except OSError:
        return set()

    window_ids = set()
    for line in result.stdout.splitlines():
        columns = line.split(maxsplit=4)
        if len(columns) < 3:
            continue
        window_class = columns[2].lower().split(".")
        if "anki" in window_class:
            window_ids.add(columns[0])
    return window_ids


def minimize_new_anki_windows(existing_window_ids):
    new_window_ids = get_anki_window_ids() - set(existing_window_ids)
    minimized = False
    for window_id in new_window_ids:
        minimized = iconify_x11_window(window_id) or minimized
    return minimized


def iconify_x11_window(window_id):
    if not os.environ.get("DISPLAY"):
        return False

    try:
        x11 = ctypes.CDLL("libX11.so.6")
        numeric_window_id = int(window_id, 16)
    except (OSError, ValueError):
        return False

    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XDefaultScreen.argtypes = [ctypes.c_void_p]
    x11.XDefaultScreen.restype = ctypes.c_int
    x11.XIconifyWindow.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
    ]
    x11.XIconifyWindow.restype = ctypes.c_int
    x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]

    display = x11.XOpenDisplay(None)
    if not display:
        return False

    try:
        screen = x11.XDefaultScreen(display)
        status = x11.XIconifyWindow(
            display,
            numeric_window_id,
            screen)
        x11.XSync(display, False)
        return status != 0
    finally:
        x11.XCloseDisplay(display)


def ensure_anki_running(
        client,
        *,
        launch=launch_anki,
        get_window_ids=get_anki_window_ids,
        minimize_new_windows=minimize_new_anki_windows,
        timeout=STARTUP_TIMEOUT_SECONDS,
        poll_interval=POLL_INTERVAL_SECONDS,
        sleep=time.sleep):
    if client.is_available():
        return False

    existing_window_ids = get_window_ids()
    launch()
    window_minimized = False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not window_minimized:
            window_minimized = minimize_new_windows(
                existing_window_ids)
        if client.is_available():
            return True
        sleep(poll_interval)

    raise AnkiConnectUnavailableError(
        "Anki started, but AnkiConnect did not become available. Install "
        f"the AnkiConnect add-on ({ANKI_CONNECT_ADDON_CODE}), restart Anki, "
        "and make sure a profile is open.")


def wait_for_collection_ready(
        client,
        *,
        timeout=COLLECTION_READY_TIMEOUT_SECONDS,
        poll_interval=POLL_INTERVAL_SECONDS,
        sleep=time.sleep,
        monotonic=time.monotonic):
    deadline = monotonic() + timeout
    last_error = None

    while monotonic() < deadline:
        try:
            return client.invoke("deckNames")
        except AnkiConnectUnavailableError as error:
            last_error = error
        except AnkiConnectResponseError as error:
            message = str(error).lower()
            if not any(
                    state in message
                    for state in ("busy", "collection", "profile", "sync")):
                raise
            last_error = error
        sleep(poll_interval)

    raise AnkiConnectUnavailableError(
        "Anki did not finish opening and synchronizing its collection "
        "within the allowed time.") from last_error


def _deck_search_query(deck_name):
    escaped_name = deck_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'deck:"{escaped_name}"'


def move_all_cards(
        client,
        *,
        source_deck,
        target_deck,
        retry_count=MOVE_RETRY_COUNT,
        retry_delay=MOVE_RETRY_DELAY_SECONDS,
        sleep=time.sleep):
    source_query = _deck_search_query(source_deck)
    remaining_card_ids = client.invoke(
        "findCards",
        query=source_query)
    moved_card_ids = set()

    for attempt in range(retry_count):
        if not remaining_card_ids:
            return len(moved_card_ids)

        client.invoke(
            "changeDeck",
            cards=remaining_card_ids,
            deck=target_deck)
        moved_card_ids.update(remaining_card_ids)
        remaining_card_ids = client.invoke(
            "findCards",
            query=source_query)

        if remaining_card_ids and attempt < retry_count - 1:
            sleep(retry_delay)

    raise AnkiIntegrationError(
        f"{len(remaining_card_ids)} cards could not be moved to "
        f'"{target_deck}". The temporary deck was kept to prevent card loss.')


def import_generated_deck(
        package_path,
        *,
        target_deck=TARGET_DECK_NAME,
        source_deck=templates.DECK_NAME,
        client=None,
        ensure_running=ensure_anki_running,
        wait_until_ready=wait_for_collection_ready):
    package_path = Path(package_path).resolve()
    if not package_path.is_file():
        raise AnkiIntegrationError(
            f"The generated Anki package does not exist: {package_path}")
    if target_deck == source_deck:
        raise AnkiIntegrationError(
            "The destination deck must differ from the generated deck.")

    client = client or AnkiConnectClient()
    ensure_running(client)

    deck_names = wait_until_ready(client)
    if target_deck not in deck_names:
        raise AnkiIntegrationError(
            f'The destination deck "{target_deck}" does not exist in Anki.')

    imported = client.invoke("importPackage", path=str(package_path))
    if imported is not True:
        raise AnkiIntegrationError(
            "AnkiConnect reported that package import failed.")

    cards_moved = move_all_cards(
        client,
        source_deck=source_deck,
        target_deck=target_deck)

    if source_deck in client.invoke("deckNames"):
        client.invoke(
            "deleteDecks",
            decks=[source_deck],
            cardsToo=True)

    return AnkiImportResult(
        cards_moved=cards_moved,
        target_deck=target_deck)
