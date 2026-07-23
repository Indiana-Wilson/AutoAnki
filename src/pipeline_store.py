import json
import os
import re
import secrets
import tempfile
import uuid
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import credential_store
import templates


PIPELINE_CONFIG_VERSION = 5
PIPELINE_CONFIG_FILE_NAME = "pipelines.json"
ANKI_DECK_CACHE_FILE_NAME = "anki_decks.json"
DEFAULT_PIPELINE_ID = "default-english-vocabulary"
DEFAULT_LANGUAGE_KEY = "english"
DEFAULT_TARGET_DECK = (
    "Retained Information::English::Modern English")
PIPELINE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class PromptOption:
    key: str
    name: str
    path: Path


@dataclass(frozen=True)
class LanguageOption:
    key: str
    name: str
    simple_prompt_key: str
    detailed_prompt_key: str
    detailed_card_type_keys: tuple[str, ...]
    card_types: tuple[tuple[str, str], ...]


LANGUAGES = (
    LanguageOption(
        key="english",
        name="English",
        simple_prompt_key="english_vocab_simple",
        detailed_prompt_key="english_vocab",
        detailed_card_type_keys=(
            templates.ENGLISH_VOCABULARY_CARD_TYPE.key,),
        card_types=(
            (
                templates.ENGLISH_VOCABULARY_CARD_TYPE.key,
                "Context sentence → meaning"),
            (
                templates.ENGLISH_WORD_TO_MEANING_CARD_TYPE.key,
                "Word → meaning"),
            (
                templates.ENGLISH_MEANING_TO_WORD_CARD_TYPE.key,
                "Meaning → word"),
        )),
    LanguageOption(
        key="classical_chinese",
        name="Classical Chinese",
        simple_prompt_key="classical_chinese_simple",
        detailed_prompt_key="classical_chinese",
        detailed_card_type_keys=(
            templates.CLASSICAL_CHINESE_CARD_TYPE.key,
            templates.CLASSICAL_CHINESE_NATIVE_VOCABULARY_CARD_TYPE.key),
        card_types=(
            (
                templates.CLASSICAL_CHINESE_CARD_TYPE.key,
                "Context sentence → English definition"),
            (
                templates.CLASSICAL_CHINESE_WORD_TO_MEANING_CARD_TYPE.key,
                "Word → English definition"),
            (
                templates.CLASSICAL_CHINESE_MEANING_TO_WORD_CARD_TYPE.key,
                "English definition → word"),
            (
                templates.CLASSICAL_CHINESE_NATIVE_VOCABULARY_CARD_TYPE.key,
                "Context sentence → native definition"),
            (
                templates
                .CLASSICAL_CHINESE_WORD_TO_NATIVE_MEANING_CARD_TYPE.key,
                "Word → native definition"),
            (
                templates
                .CLASSICAL_CHINESE_NATIVE_MEANING_TO_WORD_CARD_TYPE.key,
                "Native definition → word"),
        )),
    LanguageOption(
        key="french",
        name="French",
        simple_prompt_key="french_vocab_simple",
        detailed_prompt_key="french_vocab",
        detailed_card_type_keys=(
            templates.FRENCH_VOCABULARY_CARD_TYPE.key,
            templates.FRENCH_NATIVE_VOCABULARY_CARD_TYPE.key),
        card_types=(
            (
                templates.FRENCH_VOCABULARY_CARD_TYPE.key,
                "Context sentence → English definition"),
            (
                templates.FRENCH_WORD_TO_MEANING_CARD_TYPE.key,
                "Word → English definition"),
            (
                templates.FRENCH_MEANING_TO_WORD_CARD_TYPE.key,
                "English definition → word"),
            (
                templates.FRENCH_NATIVE_VOCABULARY_CARD_TYPE.key,
                "Context sentence → native definition"),
            (
                templates.FRENCH_WORD_TO_NATIVE_MEANING_CARD_TYPE.key,
                "Word → native definition"),
            (
                templates.FRENCH_NATIVE_MEANING_TO_WORD_CARD_TYPE.key,
                "Native definition → word"),
        )),
    LanguageOption(
        key="japanese",
        name="Japanese",
        simple_prompt_key="japanese_vocab_simple",
        detailed_prompt_key="japanese_vocab",
        detailed_card_type_keys=(
            templates.JAPANESE_VOCABULARY_CARD_TYPE.key,
            templates.JAPANESE_NATIVE_VOCABULARY_CARD_TYPE.key),
        card_types=(
            (
                templates.JAPANESE_VOCABULARY_CARD_TYPE.key,
                "Context sentence → English definition"),
            (
                templates.JAPANESE_WORD_TO_MEANING_CARD_TYPE.key,
                "Word → English definition"),
            (
                templates.JAPANESE_MEANING_TO_WORD_CARD_TYPE.key,
                "English definition → word"),
            (
                templates.JAPANESE_NATIVE_VOCABULARY_CARD_TYPE.key,
                "Context sentence → native definition"),
            (
                templates.JAPANESE_WORD_TO_NATIVE_MEANING_CARD_TYPE.key,
                "Word → native definition"),
            (
                templates.JAPANESE_NATIVE_MEANING_TO_WORD_CARD_TYPE.key,
                "Native definition → word"),
        )),
    LanguageOption(
        key="latin",
        name="Latin",
        simple_prompt_key="latin_vocab_simple",
        detailed_prompt_key="latin_vocab",
        detailed_card_type_keys=(
            templates.LATIN_VOCABULARY_CARD_TYPE.key,
            templates.LATIN_NATIVE_VOCABULARY_CARD_TYPE.key),
        card_types=(
            (
                templates.LATIN_VOCABULARY_CARD_TYPE.key,
                "Context sentence → English definition"),
            (
                templates.LATIN_WORD_TO_MEANING_CARD_TYPE.key,
                "Word → English definition"),
            (
                templates.LATIN_MEANING_TO_WORD_CARD_TYPE.key,
                "English definition → word"),
            (
                templates.LATIN_NATIVE_VOCABULARY_CARD_TYPE.key,
                "Context sentence → native definition"),
            (
                templates.LATIN_WORD_TO_NATIVE_MEANING_CARD_TYPE.key,
                "Word → native definition"),
            (
                templates.LATIN_NATIVE_MEANING_TO_WORD_CARD_TYPE.key,
                "Native definition → word"),
        )),
)


@dataclass(frozen=True)
class LanguageSettings:
    language_key: str
    card_type_keys: tuple[str, ...]
    target_deck: str
    separate_target_decks: bool = False
    card_type_target_decks: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class PipelineConfig:
    pipeline_id: str
    language_key: str
    card_type_keys: tuple[str, ...]
    target_deck: str
    generated_deck_id: int
    generated_deck_name: str
    separate_target_decks: bool = False
    card_type_target_decks: tuple[tuple[str, str], ...] = ()
    language_settings: tuple[LanguageSettings, ...] = ()


def default_pipeline():
    return PipelineConfig(
        pipeline_id=DEFAULT_PIPELINE_ID,
        language_key=DEFAULT_LANGUAGE_KEY,
        card_type_keys=(templates.DEFAULT_CARD_TYPE_KEY,),
        target_deck=DEFAULT_TARGET_DECK,
        generated_deck_id=templates.DECK_ID,
        generated_deck_name=templates.DECK_NAME)


def get_pipeline_config_path():
    return (
        credential_store.get_config_directory()
        / PIPELINE_CONFIG_FILE_NAME)


def get_anki_deck_cache_path():
    return (
        credential_store.get_config_directory()
        / ANKI_DECK_CACHE_FILE_NAME)


def load_anki_deck_cache(path=None):
    path = Path(path or get_anki_deck_cache_path())
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ()
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Cached Anki decks are not valid JSON: {path}") from error
    if (
            not isinstance(data, list)
            or not all(
                isinstance(deck_name, str) and deck_name.strip()
                for deck_name in data)):
        raise ValueError(
            "Cached Anki decks must be a list of deck names.")
    return tuple(sorted(set(data)))


def save_anki_deck_cache(deck_names, path=None):
    deck_names = tuple(sorted({
        str(deck_name).strip()
        for deck_name in deck_names
        if str(deck_name).strip()
    }))
    path = Path(path or get_anki_deck_cache_path())
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{ANKI_DECK_CACHE_FILE_NAME}.",
        dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            file_descriptor = None
            json.dump(deck_names, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    except Exception:
        if file_descriptor is not None:
            os.close(file_descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
    return deck_names


def discover_prompts(project_root):
    project_root = Path(project_root)
    prompts = []

    prompt_directory = project_root / "input" / "prompts"
    if prompt_directory.is_dir():
        for prompt_path in sorted(prompt_directory.rglob("*")):
            if not prompt_path.is_file() or prompt_path.name.startswith("."):
                continue
            relative_path = prompt_path.relative_to(prompt_directory)
            prompts.append(PromptOption(
                key=relative_path.as_posix(),
                name=prompt_path.stem.replace("_", " ").title(),
                path=prompt_path))

    return tuple(prompts)


def prompt_map(project_root):
    return {
        prompt.key: prompt
        for prompt in discover_prompts(project_root)
    }


def save_prompt_text(prompt_path, text, project_root):
    """Atomically replace one prompt inside the project's prompt directory."""
    prompt_path = Path(prompt_path).resolve()
    prompt_directory = (
        Path(project_root)
        / "input"
        / "prompts").resolve()
    try:
        prompt_path.relative_to(prompt_directory)
    except ValueError as error:
        raise ValueError(
            "Prompt files must remain inside input/prompts.") from error
    if not prompt_path.is_file():
        raise ValueError(
            f"The prompt file does not exist: {prompt_path}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A prompt cannot be empty.")

    original_mode = prompt_path.stat().st_mode & 0o777
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{prompt_path.name}.",
        dir=prompt_path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            file_descriptor = None
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary_path, original_mode)
        temporary_path.replace(prompt_path)
    except Exception:
        if file_descriptor is not None:
            os.close(file_descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
    return prompt_path


def list_languages():
    return LANGUAGES


def get_language(language_key):
    for language in LANGUAGES:
        if language.key == language_key:
            return language
    raise ValueError(f"Unknown language: {language_key}")


def get_prompt_key(pipeline):
    language = get_language(pipeline.language_key)
    if any(
            card_type_key in pipeline.card_type_keys
            for card_type_key in language.detailed_card_type_keys):
        return language.detailed_prompt_key
    return language.simple_prompt_key


def get_language_settings(pipeline, language_key):
    """Return retained preferences for one language.

    The top-level fields remain the source of truth for the active language
    so older callers that use dataclasses.replace() continue to work.
    """
    language = get_language(language_key)
    if language_key == pipeline.language_key:
        return LanguageSettings(
            language_key=language_key,
            card_type_keys=pipeline.card_type_keys,
            target_deck=pipeline.target_deck,
            separate_target_decks=pipeline.separate_target_decks,
            card_type_target_decks=pipeline.card_type_target_decks)
    for settings in pipeline.language_settings:
        if settings.language_key == language_key:
            return settings
    return LanguageSettings(
        language_key=language_key,
        card_type_keys=(language.detailed_card_type_keys[0],),
        target_deck=pipeline.target_deck)


def get_card_type_target_decks(pipeline):
    if not pipeline.separate_target_decks:
        return {
            card_type_key: pipeline.target_deck
            for card_type_key in pipeline.card_type_keys
        }
    configured_decks = dict(pipeline.card_type_target_decks)
    return {
        card_type_key: configured_decks[card_type_key]
        for card_type_key in pipeline.card_type_keys
    }


def get_model_target_decks(pipeline):
    return tuple(
        (
            templates.get_card_type(card_type_key).model.name,
            target_deck,
        )
        for card_type_key, target_deck
        in get_card_type_target_decks(pipeline).items()
    )


def create_pipeline(existing_pipelines=()):
    existing_pipelines = tuple(existing_pipelines)
    pipeline_ids = {
        pipeline.pipeline_id
        for pipeline in existing_pipelines
    }
    generated_deck_ids = {
        pipeline.generated_deck_id
        for pipeline in existing_pipelines
    }

    pipeline_id = uuid.uuid4().hex
    while pipeline_id in pipeline_ids:
        pipeline_id = uuid.uuid4().hex

    generated_deck_id = secrets.randbelow(1 << 30) + (1 << 30)
    while generated_deck_id in generated_deck_ids:
        generated_deck_id = secrets.randbelow(1 << 30) + (1 << 30)

    return PipelineConfig(
        pipeline_id=pipeline_id,
        language_key=DEFAULT_LANGUAGE_KEY,
        card_type_keys=(templates.DEFAULT_CARD_TYPE_KEY,),
        target_deck=DEFAULT_TARGET_DECK,
        generated_deck_id=generated_deck_id,
        generated_deck_name=(
            f"AutoAnki Generated - {pipeline_id[:8]}"))


def validate_pipelines(pipelines):
    pipelines = tuple(pipelines)
    pipeline_ids = set()
    generated_deck_ids = set()
    generated_deck_names = set()

    for pipeline in pipelines:
        if not pipeline.pipeline_id:
            raise ValueError("Every pipeline must have an ID.")
        if not PIPELINE_ID_PATTERN.fullmatch(pipeline.pipeline_id):
            raise ValueError(
                "Pipeline IDs may contain only letters, numbers, "
                "underscores, and hyphens.")
        if pipeline.pipeline_id in pipeline_ids:
            raise ValueError("Pipeline IDs must be unique.")
        pipeline_ids.add(pipeline.pipeline_id)

        language = get_language(pipeline.language_key)
        language_setting_keys = [
            settings.language_key
            for settings in pipeline.language_settings
        ]
        if len(language_setting_keys) != len(set(language_setting_keys)):
            raise ValueError(
                "Each language may have only one saved preference.")
        for settings in pipeline.language_settings:
            settings_language = get_language(settings.language_key)
            settings_allowed_keys = {
                card_type_key
                for card_type_key, _label
                in settings_language.card_types
            }
            if len(settings.card_type_keys) != len(
                    set(settings.card_type_keys)):
                raise ValueError(
                    "Saved card outputs may only be selected once.")
            if any(
                    card_type_key not in settings_allowed_keys
                    for card_type_key in settings.card_type_keys):
                raise ValueError(
                    f"Saved card outputs are invalid for "
                    f"{settings_language.name}.")
            if not isinstance(settings.separate_target_decks, bool):
                raise ValueError(
                    "Saved separate-target-decks must be true or false.")
            try:
                settings_target_decks = tuple(
                    (card_type_key, target_deck)
                    for card_type_key, target_deck
                    in settings.card_type_target_decks)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Saved per-card target decks have an invalid "
                    "structure.") from error
            if len(settings_target_decks) != len({
                    card_type_key
                    for card_type_key, _target_deck
                    in settings_target_decks}):
                raise ValueError(
                    "Each saved card output may have only one target deck.")
            for card_type_key, target_deck in settings_target_decks:
                if card_type_key not in settings_allowed_keys:
                    raise ValueError(
                        f"Saved per-card deck output is invalid for "
                        f"{settings_language.name}.")
                if (
                        not isinstance(target_deck, str)
                        or not target_deck.strip()):
                    raise ValueError(
                        "Every saved per-card target deck must have a name.")

        if not pipeline.card_type_keys:
            raise ValueError(
                "Select at least one card output for every pipeline.")
        if len(pipeline.card_type_keys) != len(set(pipeline.card_type_keys)):
            raise ValueError(
                "Card outputs may only be selected once per pipeline.")
        allowed_card_type_keys = {
            card_type_key
            for card_type_key, _label in language.card_types
        }
        for card_type_key in pipeline.card_type_keys:
            templates.get_card_type(card_type_key)
            if card_type_key not in allowed_card_type_keys:
                raise ValueError(
                    f'Card output "{card_type_key}" is not available for '
                    f"{language.name}.")
        if not isinstance(pipeline.separate_target_decks, bool):
            raise ValueError(
                "Separate-target-decks must be true or false.")
        try:
            card_type_target_decks = tuple(
                (card_type_key, target_deck)
                for card_type_key, target_deck
                in pipeline.card_type_target_decks)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Per-card target decks have an invalid structure.") from error
        if len(card_type_target_decks) != len({
                card_type_key
                for card_type_key, _target_deck
                in card_type_target_decks}):
            raise ValueError(
                "Each card output may have only one target deck.")
        for card_type_key, target_deck in card_type_target_decks:
            if card_type_key not in allowed_card_type_keys:
                raise ValueError(
                    f'Per-card deck output "{card_type_key}" is not '
                    f"available for {language.name}.")
            if not isinstance(target_deck, str) or not target_deck.strip():
                raise ValueError(
                    "Every configured per-card target deck must have a name.")
        if (
                not pipeline.separate_target_decks
                and not pipeline.target_deck.strip()):
            raise ValueError(
                "Every pipeline must select a target Anki deck.")
        if pipeline.separate_target_decks:
            configured_decks = dict(card_type_target_decks)
            missing_decks = [
                card_type_key
                for card_type_key in pipeline.card_type_keys
                if not configured_decks.get(card_type_key, "").strip()
            ]
            if missing_decks:
                raise ValueError(
                    "Select a target Anki deck for every selected card "
                    "output.")
        if not (1 << 30) <= pipeline.generated_deck_id < (1 << 31):
            raise ValueError(
                "Generated deck IDs must be between 2^30 and 2^31.")
        if pipeline.generated_deck_id in generated_deck_ids:
            raise ValueError("Generated deck IDs must be unique.")
        generated_deck_ids.add(pipeline.generated_deck_id)
        if not pipeline.generated_deck_name.strip():
            raise ValueError(
                "Every pipeline must have a generated deck name.")
        if pipeline.generated_deck_name in generated_deck_names:
            raise ValueError(
                "Generated deck names must be unique.")
        generated_deck_names.add(
            pipeline.generated_deck_name)

    return pipelines


def load_pipelines(path=None):
    path = Path(path or get_pipeline_config_path())
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return (default_pipeline(),)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Pipeline settings are not valid JSON: {path}") from error

    if not isinstance(data, dict):
        raise ValueError(
            "Pipeline settings must contain a JSON object.")
    version = data.get("version")
    if version not in (1, 2, 3, 4, PIPELINE_CONFIG_VERSION):
        raise ValueError(
            "Unsupported pipeline settings version.")
    try:
        pipeline_items = data["pipelines"]
        if version in (1, 2):
            pipeline_items = tuple(
                _migrate_legacy_pipeline(item)
                for item in pipeline_items)
        pipelines = tuple(
            PipelineConfig(
                **{
                    **item,
                    "card_type_keys": tuple(item["card_type_keys"]),
                    "card_type_target_decks": tuple(
                        tuple(pair)
                        for pair in item.get(
                            "card_type_target_decks",
                            ())),
                    "language_settings": tuple(
                        LanguageSettings(
                            **{
                                **settings,
                                "card_type_keys": tuple(
                                    settings["card_type_keys"]),
                                "card_type_target_decks": tuple(
                                    tuple(pair)
                                    for pair in settings.get(
                                        "card_type_target_decks",
                                        ())),
                            })
                        for settings in item.get(
                            "language_settings",
                            ())),
                })
            for item in pipeline_items)
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Pipeline settings have an invalid structure.") from error
    return validate_pipelines(pipelines)


def _migrate_legacy_pipeline(item):
    item = dict(item)
    card_type_key = item.pop(
        "card_type_key",
        templates.DEFAULT_CARD_TYPE_KEY)
    item.pop("prompt_key", None)
    if card_type_key == "basic_qa":
        card_type_key = templates.DEFAULT_CARD_TYPE_KEY

    if card_type_key == templates.CLASSICAL_CHINESE_CARD_TYPE.key:
        item["language_key"] = "classical_chinese"
    else:
        item["language_key"] = DEFAULT_LANGUAGE_KEY
    item["card_type_keys"] = (card_type_key,)
    return item


def save_pipelines(pipelines, path=None):
    pipelines = validate_pipelines(pipelines)
    path = Path(path or get_pipeline_config_path())
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    data = {
        "version": PIPELINE_CONFIG_VERSION,
        "pipelines": [
            asdict(pipeline)
            for pipeline in pipelines
        ],
    }
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{PIPELINE_CONFIG_FILE_NAME}.",
        dir=path.parent)
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            file_descriptor = None
            json.dump(data, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    except Exception:
        if file_descriptor is not None:
            os.close(file_descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
