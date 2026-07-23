import os
import sys
import tempfile
from pathlib import Path


APP_DIRECTORY_NAME = "autoanki"
API_KEY_FILE_NAME = "openai_api_key"
CONFIG_DIRECTORY_ENVIRONMENT_VARIABLE = "AUTOANKI_CONFIG_DIR"


def get_config_directory():
    override = os.environ.get(CONFIG_DIRECTORY_ENVIRONMENT_VARIABLE)
    if override:
        return Path(override).expanduser()

    if sys.platform == "win32":
        base_directory = os.environ.get("APPDATA")
        if base_directory:
            return Path(base_directory) / "AutoAnki"
        return Path.home() / "AppData" / "Roaming" / "AutoAnki"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AutoAnki"

    base_directory = os.environ.get("XDG_CONFIG_HOME")
    if base_directory:
        return Path(base_directory) / APP_DIRECTORY_NAME
    return Path.home() / ".config" / APP_DIRECTORY_NAME


def get_api_key_path():
    return get_config_directory() / API_KEY_FILE_NAME


def load_api_key():
    api_key_path = get_api_key_path()
    try:
        api_key = api_key_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return api_key or None


def save_api_key(api_key):
    api_key = api_key.strip()
    if not api_key:
        raise ValueError("The API key cannot be empty.")

    config_directory = get_config_directory()
    config_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        config_directory.chmod(0o700)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{API_KEY_FILE_NAME}.",
        dir=config_directory)
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            file_descriptor = None
            file.write(api_key)
            file.flush()
            os.fsync(file.fileno())
        if os.name == "posix":
            temporary_path.chmod(0o600)
        temporary_path.replace(get_api_key_path())
        if os.name == "posix":
            get_api_key_path().chmod(0o600)
    except Exception:
        if file_descriptor is not None:
            os.close(file_descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


def delete_api_key():
    api_key_path = get_api_key_path()
    try:
        api_key_path.unlink()
    except FileNotFoundError:
        return False
    return True
