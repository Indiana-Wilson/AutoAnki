"""Persistent configurable generation pipelines."""

import json
import os
import re
import secrets
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import credential_store
import runtime_paths
import templates


PIPELINE_CONFIG_VERSION = 7
PIPELINE_CONFIG_FILE_NAME = "pipelines.json"
ANKI_DECK_CACHE_FILE_NAME = "anki_decks.json"
DEFAULT_PIPELINE_ID = "default-english-vocabulary"
DEFAULT_LANGUAGE_KEY = "english"
DEFAULT_TARGET_DECK = "Retained Information::English::Modern English"
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
    term_field: str
    model_language_key: str


@dataclass(frozen=True)
class DirectionOption:
    key: str
    name: str
    description: str


@dataclass(frozen=True)
class FieldOption:
    key: str
    name: str


LANGUAGES = (
    LanguageOption(
        key="english",
        name="English",
        term_field="Word",
        model_language_key="english"),
    LanguageOption(
        key="classical_chinese",
        name="Classical Chinese",
        term_field="Classical Chinese",
        model_language_key="classical_chinese"),
    LanguageOption(
        key="classical_chinese_ming",
        name="Classical Chinese (Ming)",
        term_field="Classical Chinese",
        model_language_key="classical_chinese"),
    LanguageOption(
        key="classical_chinese_warring_states",
        name="Classical Chinese (Warring States)",
        term_field="Classical Chinese",
        model_language_key="classical_chinese"),
    LanguageOption(
        key="french",
        name="French",
        term_field="French",
        model_language_key="french"),
    LanguageOption(
        key="japanese",
        name="Japanese",
        term_field="Japanese",
        model_language_key="japanese"),
    LanguageOption(
        key="latin",
        name="Latin",
        term_field="Latin",
        model_language_key="latin"),
)

CARD_DIRECTIONS = (
    DirectionOption(
        key="context",
        name="Context sentence → Meaning",
        description=(
            "A varied source-language sentence on the front and the selected "
            "meaning fields on the back.")),
    DirectionOption(
        key="word_to_meaning",
        name="Word → Meaning",
        description=(
            "The word or expression on the front and the selected meaning "
            "fields on the back.")),
    DirectionOption(
        key="meaning_to_word",
        name="Meaning → Word",
        description=(
            "The selected meaning fields on the front and the word or "
            "expression on the back.")),
)

FIELD_OPTIONS = tuple(
    FieldOption(key=field_key, name=field_name)
    for field_key, field_name in templates.CONTENT_FIELDS)


@dataclass(frozen=True)
class FieldSetting:
    field_key: str
    target_language_key: str


@dataclass(frozen=True)
class CardSettings:
    direction_key: str
    enabled: bool
    fields: tuple[FieldSetting, ...]
    target_deck: str
    field_languages: tuple[FieldSetting, ...] = ()


@dataclass(frozen=True)
class LanguageSettings:
    language_key: str
    cards: tuple[CardSettings, ...]
    target_deck: str
    separate_target_decks: bool = False
    share_field_settings: bool = True
    shared_fields: tuple[FieldSetting, ...] = ()
    shared_field_languages: tuple[FieldSetting, ...] = ()


@dataclass(frozen=True)
class PipelineConfig:
    pipeline_id: str
    language_key: str
    cards: tuple[CardSettings, ...]
    target_deck: str
    generated_deck_id: int
    generated_deck_name: str
    separate_target_decks: bool = False
    share_field_settings: bool = True
    shared_fields: tuple[FieldSetting, ...] = ()
    shared_field_languages: tuple[FieldSetting, ...] = ()
    language_settings: tuple[LanguageSettings, ...] = ()


def list_languages():
    return LANGUAGES


def list_settings_languages():
    """Languages with independent Card Setup preferences."""
    return tuple(
        language
        for language in LANGUAGES
        if language.key == language.model_language_key)


def list_response_languages():
    """Languages offered for generated field content."""
    return list_settings_languages()


def list_directions():
    return CARD_DIRECTIONS


def list_field_options():
    return FIELD_OPTIONS


def get_language(language_key):
    for language in LANGUAGES:
        if language.key == language_key:
            return language
    raise ValueError(f"Unknown language: {language_key}")


def get_direction(direction_key):
    for direction in CARD_DIRECTIONS:
        if direction.key == direction_key:
            return direction
    raise ValueError(f"Unknown card direction: {direction_key}")


def get_field_option(field_key):
    for field in FIELD_OPTIONS:
        if field.key == field_key:
            return field
    raise ValueError(f"Unknown definition field: {field_key}")


def _default_fields(language_key):
    if language_key == "english":
        return (
            FieldSetting("dictionary_meaning", "english"),
            FieldSetting("pronunciation", "english"),
        )
    return (
        FieldSetting("translation", "english"),
        FieldSetting("pronunciation", "english"),
    )


def _default_field_languages(language_key):
    return tuple(
        FieldSetting(
            field.key,
            (
                "french"
                if (
                    field.key == "translation"
                    and language_key == "english")
                else "english"))
        for field in FIELD_OPTIONS)


def complete_field_languages(language_key, preferences=()):
    """Return one remembered target language for every optional field."""
    remembered = {
        field.field_key: field
        for field in _default_field_languages(language_key)}
    remembered.update({
        field.field_key: field
        for field in preferences
    })
    return tuple(
        remembered[field.key]
        for field in FIELD_OPTIONS)


def _default_language_settings(
        language_key,
        target_deck=DEFAULT_TARGET_DECK):
    fields = _default_fields(language_key)
    field_languages = complete_field_languages(
        language_key,
        fields)
    return LanguageSettings(
        language_key=language_key,
        cards=tuple(
            CardSettings(
                direction_key=direction.key,
                enabled=direction.key == "context",
                fields=fields,
                target_deck=target_deck,
                field_languages=field_languages)
            for direction in CARD_DIRECTIONS),
        target_deck=target_deck,
        share_field_settings=True,
        shared_fields=fields,
        shared_field_languages=field_languages)


def default_pipeline():
    settings = _default_language_settings(DEFAULT_LANGUAGE_KEY)
    return PipelineConfig(
        pipeline_id=DEFAULT_PIPELINE_ID,
        language_key=DEFAULT_LANGUAGE_KEY,
        cards=settings.cards,
        target_deck=settings.target_deck,
        generated_deck_id=templates.DECK_ID,
        generated_deck_name=templates.DECK_NAME,
        separate_target_decks=settings.separate_target_decks,
        share_field_settings=settings.share_field_settings,
        shared_fields=settings.shared_fields,
        shared_field_languages=settings.shared_field_languages)


def get_language_settings(pipeline, language_key):
    settings_language_key = get_language(
        language_key).model_language_key
    active_settings_language_key = get_language(
        pipeline.language_key).model_language_key
    if settings_language_key == active_settings_language_key:
        return LanguageSettings(
            language_key=settings_language_key,
            cards=pipeline.cards,
            target_deck=pipeline.target_deck,
            separate_target_decks=pipeline.separate_target_decks,
            share_field_settings=pipeline.share_field_settings,
            shared_fields=pipeline.shared_fields,
            shared_field_languages=pipeline.shared_field_languages)
    for settings in pipeline.language_settings:
        if settings.language_key == settings_language_key:
            return settings
    return _default_language_settings(
        settings_language_key,
        pipeline.target_deck)


def replace_active_language_settings(
        pipeline,
        settings,
        all_settings,
        active_language_key=None):
    active_language_key = (
        active_language_key
        or settings.language_key)
    if (
            get_language(active_language_key).model_language_key
            != settings.language_key):
        raise ValueError(
            "The generation language must use the selected settings group.")
    return replace(
        pipeline,
        language_key=active_language_key,
        cards=settings.cards,
        target_deck=settings.target_deck,
        separate_target_decks=settings.separate_target_decks,
        share_field_settings=settings.share_field_settings,
        shared_fields=settings.shared_fields,
        shared_field_languages=settings.shared_field_languages,
        language_settings=tuple(all_settings))


def get_effective_fields(settings, card):
    return (
        settings.shared_fields
        if settings.share_field_settings
        else card.fields)


def get_enabled_cards(pipeline):
    settings = get_language_settings(
        pipeline,
        pipeline.language_key)
    return tuple(
        card
        for card in settings.cards
        if card.enabled)


def get_card_type_keys(pipeline):
    model_language_key = get_language(
        pipeline.language_key).model_language_key
    return tuple(
        templates.get_direction_card_type(
            model_language_key,
            card.direction_key).key
        for card in get_enabled_cards(pipeline))


def get_card_target_decks(pipeline):
    settings = get_language_settings(
        pipeline,
        pipeline.language_key)
    return {
        card.direction_key: (
            card.target_deck
            if settings.separate_target_decks
            else settings.target_deck)
        for card in settings.cards
        if card.enabled
    }


def get_model_target_decks(pipeline):
    model_language_key = get_language(
        pipeline.language_key).model_language_key
    return tuple(
        (
            templates.get_direction_card_type(
                model_language_key,
                direction_key).model.name,
            target_deck,
        )
        for direction_key, target_deck
        in get_card_target_decks(pipeline).items())


def response_field_name(field_setting):
    field = get_field_option(field_setting.field_key)
    language = get_language(field_setting.target_language_key)
    return f"{field.name} ({language.name})"


def get_requested_field_settings(pipeline):
    settings = get_language_settings(
        pipeline,
        pipeline.language_key)
    requested = []
    seen = set()
    for card in settings.cards:
        if not card.enabled:
            continue
        for field_setting in get_effective_fields(settings, card):
            identity = (
                field_setting.field_key,
                field_setting.target_language_key)
            if identity in seen:
                continue
            seen.add(identity)
            requested.append(field_setting)
    return tuple(requested)


def requires_sentences(pipeline):
    return any(
        card.direction_key == "context"
        for card in get_enabled_cards(pipeline))


def _validate_fields(fields, source_language_key):
    fields = tuple(fields)
    field_keys = [field.field_key for field in fields]
    if len(field_keys) != len(set(field_keys)):
        raise ValueError(
            "Each definition field may be selected only once per card.")
    for field in fields:
        get_field_option(field.field_key)
        get_language(field.target_language_key)
        if (
                field.field_key == "translation"
                and field.target_language_key == source_language_key):
            source_name = get_language(source_language_key).name
            raise ValueError(
                f"Translation cannot target {source_name} when the source "
                f"language is also {source_name}.")
    return fields


def _validate_language_settings(settings):
    get_language(settings.language_key)
    if not isinstance(settings.separate_target_decks, bool):
        raise ValueError(
            "Separate-target-decks must be true or false.")
    if not isinstance(settings.share_field_settings, bool):
        raise ValueError(
            "Shared field settings must be true or false.")
    shared_fields = _validate_fields(
        settings.shared_fields,
        settings.language_key)
    _validate_fields(
        settings.shared_field_languages,
        settings.language_key)
    shared_field_languages = complete_field_languages(
            settings.language_key,
            settings.shared_field_languages)
    _validate_fields(
        shared_field_languages,
        settings.language_key)
    if len(shared_field_languages) != len(FIELD_OPTIONS):
        raise ValueError(
            "Every definition field must remember one response language.")
    direction_keys = [card.direction_key for card in settings.cards]
    expected_direction_keys = {
        direction.key
        for direction in CARD_DIRECTIONS
    }
    if (
            len(direction_keys) != len(set(direction_keys))
            or set(direction_keys) != expected_direction_keys):
        raise ValueError(
            "Every language must configure each card direction exactly once.")
    enabled_count = 0
    for card in settings.cards:
        get_direction(card.direction_key)
        if not isinstance(card.enabled, bool):
            raise ValueError("Card enabled settings must be true or false.")
        fields = _validate_fields(
            card.fields,
            settings.language_key)
        _validate_fields(
            card.field_languages,
            settings.language_key)
        field_languages = complete_field_languages(
                settings.language_key,
                card.field_languages)
        _validate_fields(
            field_languages,
            settings.language_key)
        if len(field_languages) != len(FIELD_OPTIONS):
            raise ValueError(
                "Every card field must remember one response language.")
        if not card.enabled:
            continue
        enabled_count += 1
        effective_fields = (
            shared_fields
            if settings.share_field_settings
            else fields)
        if not effective_fields:
            raise ValueError(
                "Select at least one definition field for every enabled card.")
        destination = (
            card.target_deck
            if settings.separate_target_decks
            else settings.target_deck)
        if not destination.strip():
            raise ValueError(
                "Select a target Anki deck for every enabled card.")
    if not enabled_count:
        raise ValueError("Select at least one card type.")
    return settings


def validate_pipelines(pipelines):
    pipelines = tuple(pipelines)
    pipeline_ids = set()
    generated_deck_ids = set()
    generated_deck_names = set()
    for pipeline in pipelines:
        if not PIPELINE_ID_PATTERN.fullmatch(pipeline.pipeline_id):
            raise ValueError(
                "Pipeline IDs may contain only letters, numbers, "
                "underscores, and hyphens.")
        if pipeline.pipeline_id in pipeline_ids:
            raise ValueError("Pipeline IDs must be unique.")
        pipeline_ids.add(pipeline.pipeline_id)
        active = get_language_settings(
            pipeline,
            pipeline.language_key)
        _validate_language_settings(active)
        seen_languages = set()
        for settings in pipeline.language_settings:
            if settings.language_key in seen_languages:
                raise ValueError(
                    "Each language may have only one saved preference.")
            seen_languages.add(settings.language_key)
            _validate_language_settings(settings)
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
            raise ValueError("Generated deck names must be unique.")
        generated_deck_names.add(pipeline.generated_deck_name)
    return pipelines


def create_pipeline(existing_pipelines=()):
    existing_pipelines = tuple(existing_pipelines)
    pipeline_ids = {pipeline.pipeline_id for pipeline in existing_pipelines}
    generated_ids = {
        pipeline.generated_deck_id
        for pipeline in existing_pipelines}
    pipeline_id = uuid.uuid4().hex
    while pipeline_id in pipeline_ids:
        pipeline_id = uuid.uuid4().hex
    generated_deck_id = secrets.randbelow(1 << 30) + (1 << 30)
    while generated_deck_id in generated_ids:
        generated_deck_id = secrets.randbelow(1 << 30) + (1 << 30)
    defaults = _default_language_settings(DEFAULT_LANGUAGE_KEY)
    return PipelineConfig(
        pipeline_id=pipeline_id,
        language_key=DEFAULT_LANGUAGE_KEY,
        cards=defaults.cards,
        target_deck=defaults.target_deck,
        generated_deck_id=generated_deck_id,
        generated_deck_name=f"AutoAnki Generated - {pipeline_id[:8]}",
        separate_target_decks=defaults.separate_target_decks,
        share_field_settings=defaults.share_field_settings,
        shared_fields=defaults.shared_fields,
        shared_field_languages=defaults.shared_field_languages)


def get_pipeline_config_path():
    return credential_store.get_config_directory() / PIPELINE_CONFIG_FILE_NAME


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


def _atomic_json_write(path, data, prefix):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=prefix,
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


def save_anki_deck_cache(deck_names, path=None):
    deck_names = tuple(sorted({
        str(deck_name).strip()
        for deck_name in deck_names
        if str(deck_name).strip()
    }))
    path = Path(path or get_anki_deck_cache_path())
    _atomic_json_write(
        path,
        deck_names,
        f".{ANKI_DECK_CACHE_FILE_NAME}.")
    return deck_names


def get_prompt_component_directory(project_root=None):
    if project_root is None:
        return runtime_paths.get_prompt_component_directory()
    return Path(project_root) / "input" / "prompt_components"


def discover_prompt_components(project_root=None):
    component_directory = get_prompt_component_directory(project_root)
    components = []
    if component_directory.is_dir():
        for component_path in sorted(component_directory.rglob("*")):
            if (
                    not component_path.is_file()
                    or component_path.name.startswith(".")):
                continue
            relative_path = component_path.relative_to(component_directory)
            components.append(PromptOption(
                key=relative_path.as_posix(),
                name=relative_path.as_posix().replace("_", " ").title(),
                path=component_path))
    return tuple(components)


def prompt_component_map(project_root=None):
    return {
        component.key: component
        for component in discover_prompt_components(project_root)
    }


def save_prompt_component_text(
        component_path,
        text,
        project_root=None):
    component_path = Path(component_path).resolve()
    component_directory = (
        get_prompt_component_directory(project_root).resolve())
    try:
        component_path.relative_to(component_directory)
    except ValueError as error:
        raise ValueError(
            "Prompt components must remain inside input/prompt_components."
        ) from error
    if not component_path.is_file():
        raise ValueError(
            f"The prompt component does not exist: {component_path}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A prompt component cannot be empty.")
    original_mode = component_path.stat().st_mode & 0o777
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{component_path.name}.",
        dir=component_path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            file_descriptor = None
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary_path, original_mode)
        temporary_path.replace(component_path)
    except Exception:
        if file_descriptor is not None:
            os.close(file_descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
    return component_path


# Compatibility names used by the existing Advanced editor.
discover_prompts = discover_prompt_components
prompt_map = prompt_component_map
save_prompt_text = save_prompt_component_text


def _field_from_legacy_key(card_type_key, source_language_key):
    if "native" in card_type_key:
        return FieldSetting(
            "dictionary_meaning",
            source_language_key)
    if "dictionary" in card_type_key:
        return FieldSetting("dictionary_meaning", "english")
    if source_language_key == "english":
        return FieldSetting("dictionary_meaning", "english")
    return FieldSetting("translation", "english")


def _direction_from_legacy_key(card_type_key):
    if "meaning_to_word" in card_type_key:
        return "meaning_to_word"
    if "word_to" in card_type_key:
        return "word_to_meaning"
    return "context"


def _legacy_language_settings(item, language_key, fallback_deck):
    selected_keys = tuple(item.get("card_type_keys", ()))
    if not selected_keys and item.get("card_type_key"):
        selected_keys = (item["card_type_key"],)
    if not selected_keys:
        selected_keys = (
            f"{language_key}_vocabulary",)
    per_card_decks = dict(item.get("card_type_target_decks", ()))
    fields_by_direction = {
        direction.key: []
        for direction in CARD_DIRECTIONS}
    deck_by_direction = {}
    enabled_directions = set()
    for old_key in selected_keys:
        direction_key = _direction_from_legacy_key(old_key)
        enabled_directions.add(direction_key)
        field = _field_from_legacy_key(old_key, language_key)
        if field not in fields_by_direction[direction_key]:
            fields_by_direction[direction_key].append(field)
        if direction_key == "context":
            pronunciation = FieldSetting("pronunciation", "english")
            if pronunciation not in fields_by_direction[direction_key]:
                fields_by_direction[direction_key].append(pronunciation)
        deck_by_direction.setdefault(
            direction_key,
            per_card_decks.get(old_key, fallback_deck))
    cards = tuple(
        CardSettings(
            direction_key=direction.key,
            enabled=direction.key in enabled_directions,
            fields=tuple(
                fields_by_direction[direction.key]
                or _default_fields(language_key)),
            target_deck=deck_by_direction.get(
                direction.key,
                fallback_deck),
            field_languages=complete_field_languages(
                language_key,
                tuple(
                    fields_by_direction[direction.key]
                    or _default_fields(language_key))))
        for direction in CARD_DIRECTIONS)
    selected_field_sets = {
        card.fields
        for card in cards
        if card.enabled}
    share = len(selected_field_sets) == 1
    shared_fields = (
        next(iter(selected_field_sets))
        if share and selected_field_sets
        else _default_fields(language_key))
    return LanguageSettings(
        language_key=language_key,
        cards=cards,
        target_deck=item.get("target_deck", fallback_deck),
        separate_target_decks=bool(
            item.get("separate_target_decks", False)),
        share_field_settings=share,
        shared_fields=shared_fields,
        shared_field_languages=complete_field_languages(
            language_key,
            shared_fields))


def _migrate_legacy_item(item):
    item = dict(item)
    language_key = item.get("language_key", DEFAULT_LANGUAGE_KEY)
    target_deck = item.get("target_deck", DEFAULT_TARGET_DECK)
    active = _legacy_language_settings(
        item,
        language_key,
        target_deck)
    retained = []
    for old_settings in item.get("language_settings", ()):
        retained.append(_legacy_language_settings(
            old_settings,
            old_settings["language_key"],
            old_settings.get("target_deck", target_deck)))
    return PipelineConfig(
        pipeline_id=item["pipeline_id"],
        language_key=language_key,
        cards=active.cards,
        target_deck=active.target_deck,
        generated_deck_id=item["generated_deck_id"],
        generated_deck_name=item["generated_deck_name"],
        separate_target_decks=active.separate_target_decks,
        share_field_settings=active.share_field_settings,
        shared_fields=active.shared_fields,
        shared_field_languages=active.shared_field_languages,
        language_settings=tuple(retained))


def _field_setting_from_data(data):
    return FieldSetting(
        field_key=data["field_key"],
        target_language_key=data["target_language_key"])


def _card_settings_from_data(data, language_key):
    fields = tuple(
        _field_setting_from_data(field)
        for field in data.get("fields", ()))
    saved_languages = tuple(
        _field_setting_from_data(field)
        for field in data.get("field_languages", ()))
    return CardSettings(
        direction_key=data["direction_key"],
        enabled=data["enabled"],
        fields=fields,
        target_deck=data["target_deck"],
        field_languages=complete_field_languages(
            language_key,
            saved_languages or fields))


def _language_settings_from_data(data):
    language_key = data["language_key"]
    shared_fields = tuple(
        _field_setting_from_data(field)
        for field in data.get("shared_fields", ()))
    saved_languages = tuple(
        _field_setting_from_data(field)
        for field in data.get("shared_field_languages", ()))
    return LanguageSettings(
        language_key=language_key,
        cards=tuple(
            _card_settings_from_data(card, language_key)
            for card in data["cards"]),
        target_deck=data["target_deck"],
        separate_target_decks=data.get(
            "separate_target_decks",
            False),
        share_field_settings=data.get(
            "share_field_settings",
            True),
        shared_fields=shared_fields,
        shared_field_languages=complete_field_languages(
            language_key,
            saved_languages or shared_fields))


def pipeline_to_mapping(pipeline):
    """Return the stable JSON-compatible representation of one pipeline."""
    validate_pipelines((pipeline,))
    return asdict(pipeline)


def pipeline_from_mapping(data):
    """Rebuild one current-version pipeline from retained job metadata."""
    if isinstance(data, PipelineConfig):
        validate_pipelines((data,))
        return data
    if not isinstance(data, dict):
        raise TypeError("Pipeline data must be a JSON object.")
    try:
        pipeline = PipelineConfig(
            pipeline_id=data["pipeline_id"],
            language_key=data["language_key"],
            cards=tuple(
                _card_settings_from_data(
                    card,
                    data["language_key"])
                for card in data["cards"]),
            target_deck=data["target_deck"],
            generated_deck_id=data["generated_deck_id"],
            generated_deck_name=data["generated_deck_name"],
            separate_target_decks=data.get(
                "separate_target_decks",
                False),
            share_field_settings=data.get(
                "share_field_settings",
                True),
            shared_fields=tuple(
                _field_setting_from_data(field)
                for field in data.get("shared_fields", ())),
            shared_field_languages=complete_field_languages(
                data["language_key"],
                tuple(
                    _field_setting_from_data(field)
                    for field in data.get(
                        "shared_field_languages",
                        ()))
                or tuple(
                    _field_setting_from_data(field)
                    for field in data.get("shared_fields", ()))),
            language_settings=tuple(
                _language_settings_from_data(settings)
                for settings in data.get("language_settings", ())),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Pipeline settings have an invalid structure.") from error
    validate_pipelines((pipeline,))
    return pipeline


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
    try:
        if version in (1, 2, 3, 4, 5):
            pipelines = tuple(
                _migrate_legacy_item(item)
                for item in data["pipelines"])
        elif version in (6, PIPELINE_CONFIG_VERSION):
            pipelines = tuple(
                pipeline_from_mapping(item)
                for item in data["pipelines"])
        else:
            raise ValueError(
                "Unsupported pipeline settings version.")
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Pipeline settings have an invalid structure.") from error
    return validate_pipelines(pipelines)


def save_pipelines(pipelines, path=None):
    pipelines = validate_pipelines(pipelines)
    path = Path(path or get_pipeline_config_path())
    _atomic_json_write(
        path,
        {
            "version": PIPELINE_CONFIG_VERSION,
            "pipelines": [
                asdict(pipeline)
                for pipeline in pipelines
            ],
        },
        f".{PIPELINE_CONFIG_FILE_NAME}.")
    return pipelines
