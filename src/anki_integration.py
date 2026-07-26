import base64
import ctypes
from html import unescape
from html.parser import HTMLParser
import ipaddress
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

import credential_store
import templates


ANKI_CONNECT_URL = "http://127.0.0.1:8765"
ANKI_CONNECTION_FILE_NAME = "anki_connection.json"
ANKI_CONNECT_URL_ENVIRONMENT_VARIABLE = "AUTOANKI_ANKI_CONNECT_URL"
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
class AnkiConnectionSettings:
    url: str = ANKI_CONNECT_URL
    api_key: str | None = None


@dataclass(frozen=True)
class AnkiImportResult:
    cards_moved: int
    target_deck: str
    target_decks: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnkiVocabularySource:
    """One note/card field selection used to suppress learned terms."""

    deck_name: str
    note_type_name: str
    field_name: str
    card_template_name: str | None = None


class _AnkiFieldTextExtractor(HTMLParser):
    """Reduce ordinary Anki field HTML to its visible exact text."""

    _BREAK_TAGS = frozenset({
        "br", "div", "li", "ol", "p", "table", "td", "tr", "ul",
    })

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, _attrs):
        if tag.lower() in self._BREAK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag.lower() in self._BREAK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)


def normalize_anki_connect_url(url):
    url = str(url).strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(
            "The AnkiConnect address must be an http:// or https:// URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "The AnkiConnect address cannot contain credentials, a query, "
            "or a fragment.")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError(
            "The AnkiConnect address has an invalid port.") from error
    path = parsed.path.rstrip("/")
    return urllib.parse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        path,
        "",
        "",
        "",
    ))


def is_local_anki_connect_url(url):
    hostname = urllib.parse.urlparse(url).hostname
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def get_anki_connection_path():
    return (
        credential_store.get_config_directory()
        / ANKI_CONNECTION_FILE_NAME)


def load_anki_connection_settings(path=None):
    path = Path(path or get_anki_connection_path())
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return AnkiConnectionSettings()
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Anki connection settings are not valid JSON: {path}"
        ) from error
    if not isinstance(data, dict) or set(data) - {"url", "api_key"}:
        raise ValueError(
            "Anki connection settings have an invalid structure.")
    url = normalize_anki_connect_url(data.get("url", ANKI_CONNECT_URL))
    api_key = data.get("api_key")
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError("The AnkiConnect API key must be text.")
    api_key = api_key.strip() if api_key else None
    return AnkiConnectionSettings(url=url, api_key=api_key)


def save_anki_connection_settings(url, api_key=None, path=None):
    settings = AnkiConnectionSettings(
        url=normalize_anki_connect_url(url),
        api_key=(
            (str(api_key).strip() or None)
            if api_key is not None
            else None))
    path = Path(path or get_anki_connection_path())
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        path.parent.chmod(0o700)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{ANKI_CONNECTION_FILE_NAME}.",
        dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            file_descriptor = None
            json.dump(
                {"url": settings.url, "api_key": settings.api_key},
                file,
                indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        if os.name == "posix":
            temporary_path.chmod(0o600)
        temporary_path.replace(path)
        if os.name == "posix":
            path.chmod(0o600)
    except Exception:
        if file_descriptor is not None:
            os.close(file_descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
    return settings


class AnkiConnectClient:
    def __init__(self, url=None, api_key=None, timeout=300):
        saved_settings = load_anki_connection_settings()
        configured_url = (
            url
            or os.environ.get(ANKI_CONNECT_URL_ENVIRONMENT_VARIABLE)
            or saved_settings.url)
        self.url = normalize_anki_connect_url(configured_url)
        self.api_key = (
            api_key
            if api_key is not None
            else (
                os.environ.get("ANKICONNECT_API_KEY")
                or saved_settings.api_key))
        self.timeout = timeout

    @property
    def is_local(self):
        return is_local_anki_connect_url(self.url)

    def _wrong_service_message(self, detail):
        return (
            f"The configured address {self.url} responded, but it did not "
            f"return an AnkiConnect response ({detail}). Another service may "
            "be using that port; configure AnkiConnect and AutoAnki to use "
            "the same free port.")

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
        except urllib.error.HTTPError as error:
            server = (
                error.headers.get("Server")
                if error.headers is not None
                else None)
            detail = f"HTTP {error.code}"
            if server:
                detail += f" from {server}"
            raise AnkiConnectResponseError(
                self._wrong_service_message(detail)) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise AnkiConnectResponseError(
                self._wrong_service_message("invalid JSON")) from error
        except (OSError, urllib.error.URLError) as error:
            raise AnkiConnectUnavailableError(
                "Could not connect to AnkiConnect.") from error

        if not isinstance(result, dict):
            raise AnkiConnectResponseError(
                self._wrong_service_message("invalid JSON shape"))
        if "error" not in result or "result" not in result:
            raise AnkiConnectResponseError(
                self._wrong_service_message("incomplete JSON object"))
        if result["error"] is not None:
            raise AnkiConnectResponseError(
                f"AnkiConnect error: {result['error']}")
        return result["result"]

    def is_available(self, *, raise_response_errors=False):
        try:
            version = self.invoke("version")
        except AnkiConnectResponseError:
            if raise_response_errors:
                raise
            return False
        except AnkiConnectUnavailableError:
            return False
        if isinstance(version, bool) or not isinstance(version, int):
            if raise_response_errors:
                raise AnkiConnectResponseError(
                    self._wrong_service_message(
                        "invalid version result"))
            return False
        return version >= ANKI_CONNECT_VERSION


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
    if client.is_available(raise_response_errors=True):
        return False

    if getattr(client, "is_local", True) is False:
        raise AnkiConnectUnavailableError(
            f"Could not reach remote AnkiConnect at {client.url}. Start "
            "Anki on that computer, open its profile, and check the address, "
            "API key, firewall, and AnkiConnect add-on configuration.")

    existing_window_ids = get_window_ids()
    launch()
    window_minimized = False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not window_minimized:
            window_minimized = minimize_new_windows(
                existing_window_ids)
        if client.is_available(raise_response_errors=True):
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


def _note_type_search_query(note_type_name):
    escaped_name = (
        note_type_name
        .replace("\\", "\\\\")
        .replace('"', '\\"'))
    return f'note:"{escaped_name}"'


def _card_template_search_query(card_template_name):
    escaped_name = (
        card_template_name
        .replace("\\", "\\\\")
        .replace('"', '\\"'))
    return f'card:"{escaped_name}"'


def _plain_anki_field_text(value):
    parser = _AnkiFieldTextExtractor()
    parser.feed(str(value))
    parser.close()
    return unicodedata.normalize(
        "NFC",
        " ".join(unescape("".join(parser.parts)).split()),
    )


def list_note_types_in_deck(client, deck_name):
    """Return only note types represented by at least one note in a deck."""
    note_ids = client.invoke(
        "findNotes",
        query=_deck_search_query(deck_name))
    names = set()
    for start in range(0, len(note_ids), 500):
        for note in client.invoke(
                "notesInfo",
                notes=note_ids[start:start + 500]):
            name = note.get("modelName") if isinstance(note, dict) else None
            if isinstance(name, str) and name:
                names.add(name)
    return tuple(sorted(names))


def get_note_type_fields(client, note_type_name):
    fields = client.invoke(
        "modelFieldNames",
        modelName=note_type_name)
    if not isinstance(fields, list) or not all(
            isinstance(field, str) and field
            for field in fields):
        raise AnkiConnectResponseError(
            "AnkiConnect returned invalid note-type fields.")
    return tuple(fields)


def get_note_type_card_templates(client, note_type_name):
    templates = client.invoke(
        "modelTemplates",
        modelName=note_type_name)
    if not isinstance(templates, dict):
        raise AnkiConnectResponseError(
            "AnkiConnect returned invalid card templates.")
    return tuple(sorted(
        name
        for name in templates
        if isinstance(name, str) and name))


def read_existing_vocabulary(client, selection):
    """Read exact visible terms without changing the Anki collection."""
    if not isinstance(selection, AnkiVocabularySource):
        selection = AnkiVocabularySource(**selection)
    if not all((
            selection.deck_name.strip(),
            selection.note_type_name.strip(),
            selection.field_name.strip())):
        raise ValueError(
            "Deck, note type, and field are required for vocabulary "
            "exclusion.")

    query_parts = [
        _deck_search_query(selection.deck_name),
        _note_type_search_query(selection.note_type_name),
    ]
    if selection.card_template_name:
        query_parts.append(_card_template_search_query(
            selection.card_template_name))
        card_ids = client.invoke(
            "findCards",
            query=" ".join(query_parts))
        note_ids = []
        for start in range(0, len(card_ids), 500):
            cards = client.invoke(
                "cardsInfo",
                cards=card_ids[start:start + 500])
            note_ids.extend(
                card.get("note")
                for card in cards
                if isinstance(card, dict)
                and isinstance(card.get("note"), int))
        note_ids = list(dict.fromkeys(note_ids))
    else:
        note_ids = client.invoke(
            "findNotes",
            query=" ".join(query_parts))

    vocabulary = set()
    for start in range(0, len(note_ids), 500):
        notes = client.invoke(
            "notesInfo",
            notes=note_ids[start:start + 500])
        for note in notes:
            fields = note.get("fields") if isinstance(note, dict) else None
            field = (
                fields.get(selection.field_name)
                if isinstance(fields, dict)
                else None)
            raw_value = (
                field.get("value")
                if isinstance(field, dict)
                else None)
            if not isinstance(raw_value, str):
                raise AnkiConnectResponseError(
                    f'Anki note is missing field '
                    f'"{selection.field_name}".')
            term = _plain_anki_field_text(raw_value)
            if term:
                vocabulary.add(term)
    return frozenset(vocabulary)


def import_package(client, package_path):
    """Import locally, or upload through media storage before remote import."""
    package_path = Path(package_path).resolve()
    if getattr(client, "is_local", True) is not False:
        return client.invoke("importPackage", path=str(package_path))

    remote_filename = f"_autoanki_import_{uuid.uuid4().hex}.apkg"
    encoded_package = (
        base64.b64encode(package_path.read_bytes()).decode("ascii"))
    stored_filename = client.invoke(
        "storeMediaFile",
        filename=remote_filename,
        data=encoded_package)
    if stored_filename != remote_filename:
        raise AnkiIntegrationError(
            "AnkiConnect could not stage the generated package on the "
            "remote computer.")

    try:
        media_directory = client.invoke("getMediaDirPath")
        if not isinstance(media_directory, str) or not media_directory:
            raise AnkiIntegrationError(
                "AnkiConnect did not return its remote media directory.")
        remote_path = (
            media_directory.rstrip("/\\")
            + "/"
            + remote_filename)
        imported = client.invoke("importPackage", path=remote_path)
    except Exception:
        try:
            client.invoke("deleteMediaFile", filename=remote_filename)
        except AnkiIntegrationError:
            pass
        raise

    try:
        client.invoke("deleteMediaFile", filename=remote_filename)
    except AnkiIntegrationError as error:
        raise AnkiIntegrationError(
            "The package was imported, but its temporary remote staging "
            "file could not be removed.") from error
    return imported


def import_standalone_deck(
        package_path,
        *,
        client=None,
        ensure_running=ensure_anki_running,
        wait_until_ready=wait_for_collection_ready):
    """Import a source deck in place, without moving or deleting it."""
    package_path = Path(package_path).resolve()
    if not package_path.is_file():
        raise AnkiIntegrationError(
            f"The generated Anki package does not exist: {package_path}")
    client = client or AnkiConnectClient()
    ensure_running(client)
    wait_until_ready(client)
    imported = import_package(client, package_path)
    if imported is not True:
        raise AnkiIntegrationError(
            "AnkiConnect reported that package import failed.")
    return True


def _move_matching_cards(
        client,
        *,
        source_query,
        target_deck,
        retry_count,
        retry_delay,
        sleep):
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


def move_all_cards(
        client,
        *,
        source_deck,
        target_deck,
        retry_count=MOVE_RETRY_COUNT,
        retry_delay=MOVE_RETRY_DELAY_SECONDS,
        sleep=time.sleep):
    source_query = _deck_search_query(source_deck)
    return _move_matching_cards(
        client,
        source_query=source_query,
        target_deck=target_deck,
        retry_count=retry_count,
        retry_delay=retry_delay,
        sleep=sleep)


def move_cards_by_note_type(
        client,
        *,
        source_deck,
        note_type_name,
        target_deck,
        retry_count=MOVE_RETRY_COUNT,
        retry_delay=MOVE_RETRY_DELAY_SECONDS,
        sleep=time.sleep):
    source_query = " ".join((
        _deck_search_query(source_deck),
        _note_type_search_query(note_type_name),
    ))
    return _move_matching_cards(
        client,
        source_query=source_query,
        target_deck=target_deck,
        retry_count=retry_count,
        retry_delay=retry_delay,
        sleep=sleep)


def import_generated_deck(
        package_path,
        *,
        target_deck=TARGET_DECK_NAME,
        target_decks=None,
        source_deck=templates.DECK_NAME,
        client=None,
        ensure_running=ensure_anki_running,
        wait_until_ready=wait_for_collection_ready):
    package_path = Path(package_path).resolve()
    if not package_path.is_file():
        raise AnkiIntegrationError(
            f"The generated Anki package does not exist: {package_path}")
    target_decks = (
        tuple(target_decks.items())
        if isinstance(target_decks, dict)
        else tuple(target_decks or ()))
    destination_decks = tuple(dict.fromkeys(
        (
            routed_deck
            for _note_type, routed_deck in target_decks
        )
        if target_decks
        else (target_deck,)))
    if not destination_decks:
        raise AnkiIntegrationError(
            "Select at least one destination deck.")
    if source_deck in destination_decks:
        raise AnkiIntegrationError(
            "The destination deck must differ from the generated deck.")

    client = client or AnkiConnectClient()
    ensure_running(client)

    deck_names = wait_until_ready(client)
    missing_decks = [
        deck_name
        for deck_name in destination_decks
        if deck_name not in deck_names
    ]
    if missing_decks:
        if len(missing_decks) == 1:
            raise AnkiIntegrationError(
                f'The destination deck "{missing_decks[0]}" '
                "does not exist in Anki.")
        raise AnkiIntegrationError(
            "The following destination decks do not exist in Anki: "
            + ", ".join(f'"{deck_name}"' for deck_name in missing_decks))

    imported = import_package(client, package_path)
    if imported is not True:
        raise AnkiIntegrationError(
            "AnkiConnect reported that package import failed.")

    if target_decks:
        cards_moved = sum(
            move_cards_by_note_type(
                client,
                source_deck=source_deck,
                note_type_name=note_type_name,
                target_deck=routed_deck)
            for note_type_name, routed_deck in target_decks
        )
        remaining_card_ids = client.invoke(
            "findCards",
            query=_deck_search_query(source_deck))
        if remaining_card_ids:
            raise AnkiIntegrationError(
                f"{len(remaining_card_ids)} generated cards did not match "
                "a configured card type. The temporary deck was kept to "
                "prevent card loss.")
    else:
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
        target_deck=(
            destination_decks[0]
            if len(destination_decks) == 1
            else "Multiple decks"),
        target_decks=destination_decks)
