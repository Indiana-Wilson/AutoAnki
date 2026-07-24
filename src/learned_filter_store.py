"""Persistent learned-vocabulary sources grouped by input language."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile

import credential_store
import pipeline_store


CONFIG_VERSION = 1
CONFIG_FILE_NAME = "learned_word_filters.json"


@dataclass(frozen=True)
class LearnedWordSource:
    deck_name: str = ""
    note_type: str = ""
    field_name: str = ""

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("A learned-word source must be an object.")
        return cls(
            deck_name=str(
                value.get("deck_name", value.get("deck", ""))).strip(),
            note_type=str(
                value.get("note_type", value.get("model", ""))).strip(),
            field_name=str(
                value.get("field_name", value.get("field", ""))).strip(),
        )

    @property
    def complete(self):
        return bool(
            self.deck_name
            and self.note_type
            and self.field_name)

    def to_exclusion_mapping(self):
        if not self.complete:
            raise ValueError(
                "Every learned-word source needs a deck, note type, and "
                "field.")
        return {
            "deck": self.deck_name,
            "model": self.note_type,
            "field": self.field_name,
        }


@dataclass(frozen=True)
class LanguageLearnedFilter:
    language_key: str
    enabled: bool = False
    sources: tuple[LearnedWordSource, ...] = ()

    def __post_init__(self):
        canonical = settings_language_key(self.language_key)
        object.__setattr__(self, "language_key", canonical)
        if not isinstance(self.enabled, bool):
            raise ValueError("Learned-word filtering must be enabled or off.")
        if not isinstance(self.sources, tuple):
            object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(
            self,
            "sources",
            tuple(
                LearnedWordSource.from_mapping(source)
                for source in self.sources))

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("Language learned-word settings must be an object.")
        return cls(
            language_key=value["language_key"],
            enabled=value.get("enabled", False),
            sources=tuple(
                LearnedWordSource.from_mapping(source)
                for source in value.get("sources", ())),
        )


def settings_language_key(language_key):
    """Use the same language group as Card Setup preferences."""
    return pipeline_store.get_language(language_key).model_language_key


def default_language_filter(language_key):
    return LanguageLearnedFilter(
        language_key=settings_language_key(language_key))


def source_for_new_row(sources):
    """Copy the most recent row so repeated deck setups need less typing."""
    sources = tuple(
        LearnedWordSource.from_mapping(source)
        for source in sources)
    if not sources:
        return LearnedWordSource()
    previous = sources[-1]
    return LearnedWordSource(
        deck_name=previous.deck_name,
        note_type=previous.note_type,
        field_name=previous.field_name)


def get_language_filter(settings, language_key):
    key = settings_language_key(language_key)
    for item in settings:
        item = LanguageLearnedFilter.from_mapping(item)
        if item.language_key == key:
            return item
    return default_language_filter(key)


def replace_language_filter(settings, replacement):
    replacement = LanguageLearnedFilter.from_mapping(replacement)
    by_language = {
        LanguageLearnedFilter.from_mapping(item).language_key:
            LanguageLearnedFilter.from_mapping(item)
        for item in settings
    }
    by_language[replacement.language_key] = replacement
    order = {
        language.key: index
        for index, language in enumerate(
            pipeline_store.list_settings_languages())
    }
    return tuple(sorted(
        by_language.values(),
        key=lambda item: (
            order.get(item.language_key, len(order)),
            item.language_key),
    ))


def get_config_path():
    return credential_store.get_config_directory() / CONFIG_FILE_NAME


def load_language_filters(path=None):
    path = Path(path or get_config_path())
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ()
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Learned-word settings are not valid JSON: {path}") from error
    if (
            not isinstance(value, dict)
            or value.get("version") != CONFIG_VERSION
            or not isinstance(value.get("languages"), list)):
        raise ValueError("Unsupported learned-word settings file.")
    settings = tuple(
        LanguageLearnedFilter.from_mapping(item)
        for item in value["languages"])
    keys = [item.language_key for item in settings]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "Each language may have only one learned-word configuration.")
    return settings


def save_language_filters(settings, path=None):
    settings = tuple(
        LanguageLearnedFilter.from_mapping(item)
        for item in settings)
    keys = [item.language_key for item in settings]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "Each language may have only one learned-word configuration.")
    path = Path(path or get_config_path())
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{CONFIG_FILE_NAME}.",
        dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="\n") as output:
            descriptor = None
            json.dump(
                {
                    "version": CONFIG_VERSION,
                    "languages": [
                        asdict(item)
                        for item in settings
                    ],
                },
                output,
                ensure_ascii=False,
                indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
    return settings
