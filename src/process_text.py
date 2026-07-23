import genanki
import json
import templates
import credential_store
from openai import OpenAI
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = PROJECT_ROOT / "input" / "prompts" / "english_vocab"
WORDS_PATH = PROJECT_ROOT / "input" / "words"
RESPONSE_PATH = PROJECT_ROOT / "output" / "response.json"
RESPONSE_LOG_PATH = PROJECT_ROOT / "output" / "response_log"
DECK_PATH = PROJECT_ROOT / "output" / "output.apkg"


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


def build_response_format(card_types):
    """Build the strict Structured Outputs schema for selected card types."""
    card_types = tuple(card_types)
    if not card_types:
        raise ValueError("Select at least one card type.")

    field_names = tuple(dict.fromkeys(
        field_name
        for card_type in card_types
        for field_name in card_type.field_names))
    language_fields = (
        ("Classical Chinese", "classical_chinese"),
        ("French", "french"),
        ("Japanese", "japanese"),
        ("Latin", "latin"),
        ("Word", "english"),
    )
    language_name = next(
        (
            language_name
            for field_name, language_name in language_fields
            if field_name in field_names
        ),
        "vocabulary")
    detail_name = "detailed" if "Sentences" in field_names else "simple"

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
        "name": f"autoanki_{language_name}_{detail_name}_cards",
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


def fetch_response(
        words,
        *,
        client=None,
        prompt_path=None,
        response_path=None,
        response_log_path=None,
        response_format=None):
    prompt_path = Path(prompt_path or PROMPT_PATH)
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

    prompt = prompt_path.read_text(encoding="utf-8")
    if response_format is None:
        response_format = build_response_format(
            (templates.ENGLISH_VOCABULARY_CARD_TYPE,))

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt + words,
        text={"format": response_format},
    )

    refusal = _find_refusal(response)
    if refusal:
        raise OpenAIRefusalError(refusal)

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
        raise OpenAIIncompleteResponseError(
            f"OpenAI response did not complete: {detail}")

    result = _get_response_attribute(response, "output_text", "")
    if not isinstance(result, str) or not result.strip():
        raise OpenAIIncompleteResponseError(
            "OpenAI returned no card data.")

    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_log_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(result, encoding="utf-8")

    with response_log_path.open("a", encoding="utf-8") as file:
        file.write("\n<break>\n" + result)

    return result


def fetch_response_file(file_path, **kwargs):
    words = Path(file_path).read_text(encoding="utf-8")
    return fetch_response(words, **kwargs)


# Package the notes into a deck
def process_json_text(
        text,
        *,
        deck=None,
        output_path=None,
        card_type=None,
        card_types=None,
        guid_seed=None,
        legacy_guid_card_type_key=None):
    if deck is None:
        deck = templates.my_deck
    output_path = Path(output_path or DECK_PATH)
    if card_type is not None and card_types is not None:
        raise ValueError("Pass either card_type or card_types, not both.")
    if card_types is None:
        card_types = (
            card_type or templates.ENGLISH_VOCABULARY_CARD_TYPE,)
    else:
        card_types = tuple(card_types)
    if not card_types:
        raise ValueError("Select at least one card type.")
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

    expected_fields = tuple(dict.fromkeys(
        field_name
        for selected_card_type in card_types
        for field_name in selected_card_type.field_names))
    expected_field_set = set(expected_fields)
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
        validated_notes.append(note_data)

    notes_created = 0
    for note_data in validated_notes:
        for selected_card_type in card_types:
            fields = [
                note_data[field_name]
                for field_name in selected_card_type.field_names
            ]
            note_arguments = {
                "model": selected_card_type.model,
                "fields": fields,
            }
            if (
                    guid_seed is not None
                    and selected_card_type.key
                    != legacy_guid_card_type_key):
                identity_values = [
                    note_data[field_name]
                    for field_name in selected_card_type.identity_fields
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
        card_type_key=templates.DEFAULT_CARD_TYPE_KEY,
        card_type_keys=None,
        guid_seed=None,
        legacy_guid_card_type_key=None):
    """Generate one fresh Anki deck from the supplied words."""
    response_path = Path(response_path or RESPONSE_PATH)
    response_log_path = Path(
        response_log_path or RESPONSE_LOG_PATH)
    output_path = Path(output_path or DECK_PATH)
    if card_type_keys is None:
        card_type_keys = (card_type_key,)
    card_types = tuple(
        templates.get_card_type(key)
        for key in card_type_keys)
    response_text = fetch_response(
        words,
        client=client,
        prompt_path=prompt_path,
        response_path=response_path,
        response_log_path=response_log_path,
        response_format=build_response_format(card_types))
    process_json_text(
        response_text,
        deck=templates.create_deck(deck_id, deck_name),
        output_path=output_path,
        card_types=card_types,
        guid_seed=guid_seed,
        legacy_guid_card_type_key=legacy_guid_card_type_key)
    return output_path


def main():
    words = WORDS_PATH.read_text(encoding="utf-8")
    generate_deck(words)


if __name__ == "__main__":
    main()
