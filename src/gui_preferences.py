"""Small, versioned preferences for controls not owned by pipeline settings.

Paid-action acknowledgements and edit-unlock safeguards are intentionally not
part of this schema.  They must always start disabled.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import credential_store


CONFIG_VERSION = 1
CONFIG_FILE_NAME = "gui_preferences.json"

DEFAULTS = {
    "source_key": "",
    "source_language_overrides": {},
    "source_chunk_size": "30",
    "source_limit_to_prefix": False,
    "source_prefix_token_limit": "100",
    "source_concurrency": "8",
    "source_request_stagger_ms": "100",
    "source_allow_web_search": False,
    "source_use_source_examples": False,
    "source_card_directions": ["context"],
    "source_include_context_nuance": False,
    "source_separate_decks": False,
    "source_model_key": "gpt-5.4-mini",
    "source_protocol_key": "v10",
    "source_reasoning_key": "low",
    "source_execution_key": "standard",
    "source_automatic_repair": False,
    "source_context_key": "sentence",
    "source_non_example_context_key": "sentence",
    "source_preview_page_size": "100",
    "source_file_language": "Classical Chinese (Warring States)",
    "source_file_use_gpu": True,
    "source_codex_language": "Classical Chinese (Warring States)",
    "prompt_key": "",
    "main_tab": "generate",
    "generate_tab": "manual",
    "source_tab": "generate",
}


def get_config_path():
    return credential_store.get_config_directory() / CONFIG_FILE_NAME


def default_preferences():
    return deepcopy(DEFAULTS)


def _validated_preferences(value):
    if not isinstance(value, dict):
        raise ValueError("GUI preferences must be a JSON object.")
    if value.get("version") != CONFIG_VERSION:
        raise ValueError("Unsupported GUI preferences version.")
    supplied = value.get("preferences")
    if not isinstance(supplied, dict):
        raise ValueError("GUI preferences are missing their settings object.")

    result = default_preferences()
    for key, default in DEFAULTS.items():
        if key not in supplied:
            continue
        candidate = supplied[key]
        if isinstance(default, bool):
            if type(candidate) is not bool:
                raise ValueError(f"GUI preference {key!r} must be true/false.")
        elif isinstance(default, str):
            if not isinstance(candidate, str):
                raise ValueError(f"GUI preference {key!r} must be text.")
        elif isinstance(default, list):
            if (
                    not isinstance(candidate, list)
                    or not all(isinstance(item, str) for item in candidate)):
                raise ValueError(
                    f"GUI preference {key!r} must be a list of text values.")
        elif isinstance(default, dict):
            if (
                    not isinstance(candidate, dict)
                    or not all(
                        isinstance(item_key, str)
                        and isinstance(item_value, str)
                        for item_key, item_value in candidate.items())):
                raise ValueError(
                    f"GUI preference {key!r} must map text to text.")
        result[key] = deepcopy(candidate)
    return result


def load_preferences(path=None):
    path = Path(path or get_config_path())
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default_preferences()
    except json.JSONDecodeError as error:
        raise ValueError(
            f"GUI preferences are not valid JSON: {path}") from error
    return _validated_preferences(value)


def save_preferences(preferences, path=None):
    path = Path(path or get_config_path())
    validated = _validated_preferences({
        "version": CONFIG_VERSION,
        "preferences": preferences,
    })
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        path.parent.chmod(0o700)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{CONFIG_FILE_NAME}.",
        dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            descriptor = None
            json.dump(
                {
                    "version": CONFIG_VERSION,
                    "preferences": validated,
                },
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        if os.name == "posix":
            temporary_path.chmod(0o600)
        temporary_path.replace(path)
        if os.name == "posix":
            path.chmod(0o600)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
    return validated
