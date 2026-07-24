import genanki
import json
import templates
import credential_store
import pipeline_store
import prompt_builder
import runtime_paths
from openai import OpenAI
import os
from pathlib import Path


PROJECT_ROOT = runtime_paths.get_resource_root()
PROMPT_DIRECTORY = runtime_paths.get_prompt_directory()
OUTPUT_DIRECTORY = runtime_paths.get_output_directory()
PROMPT_PATH = PROMPT_DIRECTORY / "english_vocab"
WORDS_PATH = runtime_paths.get_words_path()
RESPONSE_PATH = OUTPUT_DIRECTORY / "response.json"
RESPONSE_LOG_PATH = OUTPUT_DIRECTORY / "response_log"
DECK_PATH = OUTPUT_DIRECTORY / "output.apkg"


class MissingAPIKeyError(RuntimeError):
    pass


class OpenAIResponseError(RuntimeError):
    """The API returned a response that cannot safely be processed."""


class OpenAIRefusalError(OpenAIResponseError):
    """The model refused the generation request."""


class OpenAIIncompleteResponseError(OpenAIResponseError):
    """The API response did not complete normally."""


class GeneratedCardValidationError(ValueError):
    """Generated JSON is structurally valid but unusable as card data."""


def get_api_key():
    environment_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    return credential_store.load_api_key()


def get_response_field_names(pipeline):
    pipeline_store.validate_pipelines((pipeline,))
    language = pipeline_store.get_language(
        pipeline.language_key)
    field_names = [language.term_field]
    if pipeline_store.requires_sentences(pipeline):
        field_names.append("Sentences")
    field_names.extend(
        pipeline_store.response_field_name(field_setting)
        for field_setting
        in pipeline_store.get_requested_field_settings(pipeline))
    return tuple(field_names)


def build_response_format(pipeline):
    """Build a strict schema containing only fields selected in the UI."""
    field_names = get_response_field_names(pipeline)
    detail_name = (
        "context"
        if pipeline_store.requires_sentences(pipeline)
        else "simple")

    card_schema = {
        "type": "object",
        "properties": {
            field_name: {"type": "string"}
            for field_name in field_names
        },
        "required": list(field_names),
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "name": (
            f"autoanki_{pipeline.language_key}_{detail_name}_cards"),
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "cards": {
                    "type": "array",
                    "items": card_schema,
                },
            },
            "required": ["cards"],
            "additionalProperties": False,
        },
    }


def _get_response_attribute(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _find_refusal(response):
    for output_item in _get_response_attribute(response, "output", ()) or ():
        if _get_response_attribute(output_item, "type") != "message":
            continue
        for content_item in (
                _get_response_attribute(output_item, "content", ()) or ()):
            if _get_response_attribute(content_item, "type") == "refusal":
                return (
                    _get_response_attribute(content_item, "refusal")
                    or "The model refused the request.")
    return None


def extract_response_text(response):
    """Validate one Responses API result and return its structured JSON text.

    This intentionally performs no file writes.  Source-generation workers
    persist the raw response in their own per-attempt directories instead.
    """
    refusal = _find_refusal(response)
    if refusal:
        error = OpenAIRefusalError(refusal)
        error.raw_response_text = _get_response_attribute(
            response,
            "output_text",
            "")
        raise error

    status = _get_response_attribute(response, "status")
    if status != "completed":
        incomplete_details = _get_response_attribute(
            response,
            "incomplete_details")
        reason = _get_response_attribute(
            incomplete_details,
            "reason")
        error = _get_response_attribute(response, "error")
        detail = reason or error or status or "unknown status"
        error = OpenAIIncompleteResponseError(
            f"OpenAI response did not complete: {detail}")
        error.raw_response_text = _get_response_attribute(
            response,
            "output_text",
            "")
        raise error

    result = _get_response_attribute(response, "output_text", "")
    if not isinstance(result, str) or not result.strip():
        error = OpenAIIncompleteResponseError(
            "OpenAI returned no card data.")
        error.raw_response_text = (
            result if isinstance(result, str) else "")
        raise error
    return result


def fetch_response(
        words,
        *,
        client=None,
        prompt_path=None,
        response_path=None,
        response_log_path=None,
        response_format=None,
        prompt_text=None):
    response_path = Path(response_path or RESPONSE_PATH)
    response_log_path = Path(response_log_path or RESPONSE_LOG_PATH)

    if client is None:
        key = get_api_key()
        if not key:
            raise MissingAPIKeyError(
                "No OpenAI API key is configured. Add one through the "
                "AutoAnki GUI or set OPENAI_API_KEY.")
        # A paid request is retried only through the GUI's explicit
        # Retry/Cancel decision, never automatically by the SDK.
        client = OpenAI(api_key=key, max_retries=0)

    if prompt_text is None:
        prompt_path = Path(prompt_path or PROMPT_PATH)
        prompt_text = prompt_path.read_text(encoding="utf-8")
    if response_format is None:
        response_format = build_response_format(
            pipeline_store.default_pipeline())

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt_text + words,
        reasoning={"effort": "none"},
        text={"format": response_format},
    )

    result = extract_response_text(response)

    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_log_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(result, encoding="utf-8")

    with response_log_path.open("a", encoding="utf-8") as file:
        file.write("\n<break>\n" + result)

    return result


def fetch_response_file(file_path, **kwargs):
    words = Path(file_path).read_text(encoding="utf-8")
    return fetch_response(words, **kwargs)


def validate_generated_cards(text, pipeline=None):
    """Parse and validate generated data without writing an Anki package."""
    pipeline = pipeline or pipeline_store.default_pipeline()
    pipeline_store.validate_pipelines((pipeline,))
    language_settings = pipeline_store.get_language_settings(
        pipeline,
        pipeline.language_key)
    enabled_cards = pipeline_store.get_enabled_cards(pipeline)
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        if set(parsed) != {"cards"}:
            raise GeneratedCardValidationError(
                'The generated JSON object must contain only "cards".')
        parser = parsed["cards"]
    else:
        # Retain support for responses produced before Structured Outputs
        # wrapped the array in a root object.
        parser = parsed
    if not isinstance(parser, list):
        raise GeneratedCardValidationError(
            "Generated card data must be an array.")

    expected_fields = get_response_field_names(pipeline)
    expected_field_set = set(expected_fields)
    empty_allowed_fields = {
        pipeline_store.response_field_name(field_setting)
        for field_setting
        in pipeline_store.get_requested_field_settings(pipeline)
        if field_setting.field_key == "nuance"
    }
    validated_notes = []
    for note_data in parser:
        if not isinstance(note_data, dict):
            raise GeneratedCardValidationError(
                "Every generated card must be a JSON object.")
        if set(note_data) != expected_field_set:
            missing = expected_field_set - set(note_data)
            unexpected = set(note_data) - expected_field_set
            details = []
            if missing:
                details.append(
                    "missing " + ", ".join(sorted(missing)))
            if unexpected:
                details.append(
                    "unexpected " + ", ".join(sorted(unexpected)))
            raise GeneratedCardValidationError(
                "Generated card fields are invalid: "
                + "; ".join(details))
        for field_name in expected_fields:
            field = note_data[field_name]
            if not isinstance(field, str):
                raise TypeError(
                    "All generated card fields must be strings.")
            if not field.strip():
                if field_name in empty_allowed_fields:
                    note_data[field_name] = ""
                    continue
                raise GeneratedCardValidationError(
                    f'Generated field "{field_name}" cannot be empty.')
        if "Sentences" in expected_field_set:
            sentences = [
                sentence.strip()
                for sentence in note_data["Sentences"].split("|")
                if sentence.strip()
            ]
            if len(sentences) != 4:
                raise GeneratedCardValidationError(
                    "Each detailed card must contain exactly four "
                    "pipe-separated example sentences.")
        for card in enabled_cards:
            effective_fields = pipeline_store.get_effective_fields(
                language_settings,
                card)
            if not any(
                    note_data[
                        pipeline_store.response_field_name(
                            field_setting)].strip()
                    for field_setting in effective_fields):
                direction_name = pipeline_store.get_direction(
                    card.direction_key).name
                raise GeneratedCardValidationError(
                    f'"{direction_name}" has no non-empty meaning fields.')
        validated_notes.append(note_data)

    return validated_notes


def validate_generated_response(text, pipeline=None):
    """Return the canonical response object after structural validation."""
    return {
        "cards": validate_generated_cards(text, pipeline),
    }


# Package the notes into a deck
def process_json_text(
        text,
        *,
        deck=None,
        output_path=None,
        pipeline=None,
        guid_seed=None,
        due_start=None,
        **_legacy_arguments):
    if deck is None:
        deck = templates.my_deck
    output_path = Path(output_path or DECK_PATH)
    pipeline = pipeline or pipeline_store.default_pipeline()
    pipeline_store.validate_pipelines((pipeline,))
    language = pipeline_store.get_language(
        pipeline.language_key)
    model_language_key = language.model_language_key
    language_settings = pipeline_store.get_language_settings(
        pipeline,
        pipeline.language_key)
    enabled_cards = pipeline_store.get_enabled_cards(pipeline)
    validated_notes = validate_generated_cards(text, pipeline)
    if (
            due_start is not None
            and (
                isinstance(due_start, bool)
                or not isinstance(due_start, int)
                or due_start < 1)):
        raise ValueError(
            "Anki new-card ordering must start at a positive integer.")

    notes_created = 0
    for note_data in validated_notes:
        for card in enabled_cards:
            selected_card_type = templates.get_direction_card_type(
                model_language_key,
                card.direction_key)
            effective_fields = pipeline_store.get_effective_fields(
                language_settings,
                card)
            response_fields_by_key = {
                field_setting.field_key: (
                    pipeline_store.response_field_name(field_setting))
                for field_setting in effective_fields
            }
            fields = [
                note_data[language.term_field],
                note_data.get("Sentences", ""),
                *(
                    note_data.get(
                        response_fields_by_key.get(field_key, ""),
                        "")
                    for field_key, _field_name
                    in templates.CONTENT_FIELDS
                ),
            ]
            note_arguments = {
                "model": selected_card_type.model,
                "fields": fields,
            }
            if due_start is not None:
                # genanki's default is due=0 for every new card. Source decks
                # set a sequence so Anki's ascending-position gather order
                # follows the retained first-occurrence/card order.
                note_arguments["due"] = due_start + notes_created
            if guid_seed is not None:
                identity_values = [
                    note_data[language.term_field],
                    *(
                        note_data[
                            pipeline_store.response_field_name(field_setting)]
                        for field_setting in effective_fields),
                ]
                note_arguments["guid"] = genanki.guid_for(
                    "autoanki",
                    guid_seed,
                    selected_card_type.key,
                    *identity_values)

            deck.add_note(genanki.Note(**note_arguments))
            notes_created += 1

    # Export the notes into a deck
    output_path.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(output_path)
    print("Card deck successfully created.")
    return notes_created


def process_json_file(file_path, **kwargs):
    text = Path(file_path).read_text(encoding="utf-8")
    return process_json_text(text, **kwargs)


def generate_deck(
        words,
        *,
        client=None,
        prompt_path=None,
        response_path=None,
        response_log_path=None,
        output_path=None,
        deck_id=templates.DECK_ID,
        deck_name=templates.DECK_NAME,
        pipeline=None,
        prompt_text=None,
        guid_seed=None,
        **_legacy_arguments):
    """Generate one fresh Anki deck from the supplied words."""
    response_path = Path(response_path or RESPONSE_PATH)
    response_log_path = Path(
        response_log_path or RESPONSE_LOG_PATH)
    output_path = Path(output_path or DECK_PATH)
    pipeline = pipeline or pipeline_store.default_pipeline()
    if prompt_text is None:
        prompt_text = prompt_builder.build_prompt(pipeline)
    response_text = fetch_response(
        words,
        client=client,
        prompt_path=prompt_path,
        prompt_text=prompt_text,
        response_path=response_path,
        response_log_path=response_log_path,
        response_format=build_response_format(pipeline))
    process_json_text(
        response_text,
        deck=templates.create_deck(deck_id, deck_name),
        output_path=output_path,
        pipeline=pipeline,
        guid_seed=guid_seed,
    )
    return output_path


def main():
    words = WORDS_PATH.read_text(encoding="utf-8")
    generate_deck(words)


if __name__ == "__main__":
    main()
