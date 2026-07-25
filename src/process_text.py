import genanki
import hashlib
import html
import json
import re
import unicodedata
import templates
import credential_store
import pipeline_store
import prompt_builder
from response_schema import bounded_response_format_name
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
SENTENCE_TRANSLATIONS_FIELD_NAME = (
    templates.SENTENCE_TRANSLATIONS_FIELD_NAME)
SOURCE_CONTEXT_TRANSLATIONS_KEY = "source_context_translations"
SOURCE_CONTEXT_ID_FIELD_NAME = "context_id"
SOURCE_CONTEXT_TRANSLATION_FIELD_NAME = "translation"
_WANG_BI_TIANDI_ZHI_SHI_CONTEXTUAL_FIELDS = {
    "Translation (English)": "beginning; origin",
    "Dictionary Meaning (English)": (
        "The starting point or origin of something."),
    "Pronunciation (English)": "shǐ",
    "Part of Speech (English)": "noun",
    "Nuance (English)": (
        "Here it names the origin of heaven and earth, not the verbal act "
        "of beginning."),
}
_WANG_BI_COMPACT_CONTEXTUAL_FIELDS = {
    (
        "無名，天地之始，有名，萬物之母。",
        "有",
    ): {
        "Translation (English)": "having a name; named",
        "Pronunciation (English)": "yǒu",
        "Part of Speech (English)": "verb",
    },
    (
        "故常無欲，以觀其妙，",
        "以",
    ): {
        "Translation (English)": "in order to; so as to",
        "Pronunciation (English)": "yǐ",
        "Part of Speech (English)": "conjunction",
    },
    (
        "常有欲，以觀其徼；",
        "徼",
    ): {
        "Translation (English)": "outward manifestation; outer limit",
        "Pronunciation (English)": "jiào",
        "Part of Speech (English)": "noun",
    },
    (
        "此兩者，同出而異名，同謂之玄。",
        "者",
    ): {
        "Translation (English)": "the two (things)",
        "Pronunciation (English)": "zhě",
        "Part of Speech (English)": "nominalizing particle",
    },
    (
        "天下皆知美之為美，斯惡已。",
        "美",
    ): {
        "Translation (English)": "beauty; the beautiful",
        "Pronunciation (English)": "měi",
        "Part of Speech (English)": "noun",
    },
    (
        "天下皆知美之為美，斯惡已。",
        "為",
    ): {
        "Translation (English)": "to be regarded as; to constitute",
        "Pronunciation (English)": "wéi",
        "Part of Speech (English)": "verb",
    },
    (
        "天下皆知美之為美，斯惡已。",
        "惡",
    ): {
        "Translation (English)": "ugliness; the ugly",
        "Pronunciation (English)": "è",
        "Part of Speech (English)": "noun",
    },
    (
        "行不言之教；萬物作焉而不辭，生而不有，為而不恃，",
        "辭",
    ): {
        "Translation (English)": "to decline; to refuse",
        "Pronunciation (English)": "cí",
        "Part of Speech (English)": "verb",
    },
    (
        "功成而弗居。",
        "居",
    ): {
        "Translation (English)": "to claim or appropriate credit",
        "Pronunciation (English)": "jū",
        "Part of Speech (English)": "verb",
    },
    (
        "夫唯弗居，是以不去。",
        "夫",
    ): {
        "Translation (English)": "now; indeed; as for",
        "Pronunciation (English)": "fú",
        "Part of Speech (English)": "discourse particle",
    },
    (
        "不尚賢，使民不爭；不貴難得之貨，使民不為盜；"
        "不見可欲，使民心不亂。",
        "見",
    ): {
        "Translation (English)": "to display; to show",
        "Pronunciation (English)": "xiàn",
        "Part of Speech (English)": "verb",
    },
    (
        "不尚賢，使民不爭；不貴難得之貨，使民不為盜；"
        "不見可欲，使民心不亂。",
        "民心",
    ): {
        "Translation (English)": "the people's hearts and minds",
        "Pronunciation (English)": "mín xīn",
        "Part of Speech (English)": "noun phrase",
    },
    (
        "使夫智者不敢為也。",
        "智者",
    ): {
        "Translation (English)": "wise or clever people",
        "Pronunciation (English)": "zhì zhě",
        "Part of Speech (English)": "noun phrase",
    },
    (
        "道沖而用之或不盈，淵兮似萬物之宗；挫其銳，解其紛，"
        "和其光，同其塵，湛兮似或存。",
        "沖",
    ): {
        "Translation (English)": "empty; hollow; open",
        "Pronunciation (English)": "chōng",
        "Part of Speech (English)": "stative verb / adjective",
    },
}
SOURCE_CONTEXTUAL_CARDS_KEY = "contextual_cards"
SOURCE_ADDITIONAL_SENSE_CARDS_KEY = "additional_sense_cards"
SOURCE_TERM_RESULTS_KEY = "term_results"
SOURCE_RANK_FIELD_NAME = "rank"
SOURCE_CONTEXTUAL_SENSE_KEY = "contextual_sense"
SOURCE_ADDITIONAL_SENSES_KEY = "additional_senses"


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


_STRONG_TAG_PATTERN = re.compile(r"</?strong\b[^>]*>", re.IGNORECASE)
_STRONG_SPAN_PATTERN = re.compile(
    r"<strong\b[^>]*>(.*?)</strong\s*>",
    re.IGNORECASE | re.DOTALL)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_SOURCE_SCRIPT_PATTERNS = {
    "classical_chinese": re.compile(
        "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
        "\U00020000-\U0002fa1f]"),
    "japanese": re.compile(
        "[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f"
        "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
        "\U00020000-\U0002fa1f]"),
}
_UNBOUNDED_TERM_LANGUAGES = frozenset({
    "classical_chinese",
    "classical_chinese_han",
    "classical_chinese_ming",
    "classical_chinese_warring_states",
    "classical_chinese_wang_bi",
    "japanese",
})


def _term_occurrence_pattern(term, language_key):
    """Match one literal term occurrence without consuming its surroundings."""
    escaped = re.escape(term)
    if language_key in _UNBOUNDED_TERM_LANGUAGES:
        return re.compile(escaped, re.IGNORECASE)
    prefix = r"(?<!\w)" if term[0].isalnum() else ""
    suffix = r"(?!\w)" if term[-1].isalnum() else ""
    return re.compile(prefix + escaped + suffix, re.IGNORECASE)


def emphasize_term_in_sentences(sentences_text, term, language_key):
    """Emphasize literal occurrences in an inserted source-context passage.

    This is retained for context text added after the model response. Generated
    examples instead preserve the model's sense-aware markup through
    ``sanitize_emphasized_sentences``.
    """
    if not isinstance(sentences_text, str):
        raise TypeError("Example sentences must be text.")
    if not isinstance(term, str):
        raise TypeError("The card term must be text.")
    term = term.strip()
    sentences = sentences_text.split("|")
    if not term:
        return "|".join(
            html.escape(
                html.unescape(_STRONG_TAG_PATTERN.sub("", sentence)),
                quote=False)
            for sentence in sentences)

    pattern = _term_occurrence_pattern(term, language_key)
    emphasized = []
    for sentence in sentences:
        plain = html.unescape(_STRONG_TAG_PATTERN.sub("", sentence))
        parts = []
        cursor = 0
        for match in pattern.finditer(plain):
            parts.append(html.escape(plain[cursor:match.start()], quote=False))
            parts.append(
                "<strong>"
                + html.escape(match.group(0), quote=False)
                + "</strong>")
            cursor = match.end()
        parts.append(html.escape(plain[cursor:], quote=False))
        emphasized.append("".join(parts))
    return "|".join(emphasized)


def emphasize_source_occurrence(
        context_text,
        term,
        language_key,
        *,
        start_offset,
        end_offset):
    """Emphasize only the audited occurrence that selected this source word."""
    if not isinstance(context_text, str):
        raise TypeError("Source context must be text.")
    if not isinstance(term, str) or not term:
        raise ValueError("A non-empty source term is required.")
    if (
            isinstance(start_offset, bool)
            or not isinstance(start_offset, int)
            or isinstance(end_offset, bool)
            or not isinstance(end_offset, int)
            or start_offset < 0
            or end_offset <= start_offset
            or end_offset > len(context_text)):
        raise ValueError(
            "Source occurrence offsets must identify text inside its context.")
    occurrence = context_text[start_offset:end_offset]
    if _term_occurrence_pattern(
            term,
            language_key).fullmatch(occurrence) is None:
        raise ValueError(
            "Source occurrence offsets do not match the requested term.")
    return (
        html.escape(context_text[:start_offset], quote=False)
        + "<strong>"
        + html.escape(occurrence, quote=False)
        + "</strong>"
        + html.escape(context_text[end_offset:], quote=False))


def sanitize_emphasized_sentences(sentences_text):
    """Preserve model-selected strong spans while escaping all other HTML."""
    if not isinstance(sentences_text, str):
        raise TypeError("Example sentences must be text.")
    sanitized_sentences = []
    for sentence in sentences_text.split("|"):
        parts = []
        cursor = 0
        strong_open = False
        for match in _STRONG_TAG_PATTERN.finditer(sentence):
            parts.append(html.escape(
                html.unescape(sentence[cursor:match.start()]),
                quote=False))
            closing = match.group(0).lstrip().startswith("</")
            if closing:
                if strong_open:
                    parts.append("</strong>")
                    strong_open = False
            elif not strong_open:
                parts.append("<strong>")
                strong_open = True
            cursor = match.end()
        parts.append(html.escape(
            html.unescape(sentence[cursor:]),
            quote=False))
        if strong_open:
            parts.append("</strong>")
        sanitized_sentences.append("".join(parts))
    return "|".join(sanitized_sentences)


def sanitize_sentence_translations(translations_text):
    """Escape generated translations while preserving their pipe alignment."""
    if not isinstance(translations_text, str):
        raise TypeError("Example-sentence translations must be text.")
    return "|".join(
        html.escape(html.unescape(translation), quote=False)
        for translation in translations_text.split("|"))


def encode_source_context_block(context_text):
    """Protect literal source pipes from the generated-example delimiter."""
    if not isinstance(context_text, str):
        raise TypeError("Source context must be text.")
    marker = templates.SOURCE_CONTEXT_ESCAPE_MARKER
    encoded = (
        context_text
        .replace(marker, marker + "0")
        .replace("|", marker + "1"))
    return templates.SOURCE_CONTEXT_BLOCK_PREFIX + encoded


def _sentence_has_emphasized_usage(sentence):
    return any(
        span.strip()
        for span in _STRONG_SPAN_PATTERN.findall(sentence))


def _plain_sentence_item(value):
    """Return comparison text with sense-marking strong tags removed."""
    return html.unescape(_STRONG_TAG_PATTERN.sub("", value)).strip()


def _has_latin_alphabetic_text(value):
    return any(
        character.isalpha()
        and "LATIN" in unicodedata.name(character, "")
        for character in value)


def _sentence_translation_language_issue(
        source_sentence,
        translation,
        source_model_language_key):
    """Classify only high-confidence failures of an English translation."""
    plain_source = _plain_sentence_item(source_sentence)
    plain_translation = _plain_sentence_item(translation)
    if plain_source and plain_translation == plain_source:
        return "exact_source_copy", plain_source

    source_script_pattern = _SOURCE_SCRIPT_PATTERNS.get(
        source_model_language_key)
    if (
            plain_translation
            and source_script_pattern is not None
            and source_script_pattern.search(plain_translation)
            and not _has_latin_alphabetic_text(plain_translation)):
        return "source_script_without_latin_text", plain_source
    return None, plain_source


def _generated_problem_id(code, path, identity=None):
    """Return a deterministic identity for one response-validation problem."""
    encoded = json.dumps(
        {
            "code": code,
            "path": path,
            "identity": identity,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{code}:{hashlib.sha256(encoded).hexdigest()[:16]}"


def _bounded_problem_text(value, limit):
    text = str(value)
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"… [{omitted:,} more characters]"


def _bounded_problem_value(value, *, depth=0):
    """Keep diagnostics JSON-safe and small enough for an inspection UI."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _bounded_problem_text(value, 500)
    if depth >= 3:
        return _bounded_problem_text(value, 500)
    if isinstance(value, (tuple, list, set, frozenset)):
        values = list(value)
        bounded = [
            _bounded_problem_value(item, depth=depth + 1)
            for item in values[:50]
        ]
        if len(values) > 50:
            bounded.append(f"… [{len(values) - 50:,} more items]")
        return bounded
    if isinstance(value, dict):
        items = list(value.items())
        bounded = {
            _bounded_problem_text(key, 100): _bounded_problem_value(
                item,
                depth=depth + 1)
            for key, item in items[:50]
        }
        if len(items) > 50:
            bounded["…"] = f"[{len(items) - 50:,} more entries]"
        return bounded
    return _bounded_problem_text(value, 500)


def _generated_problem(
        code,
        title,
        message,
        *,
        path="$",
        location="Response",
        scope="response",
        overrideable=False,
        card_index=None,
        term=None,
        field_name=None,
        expected=None,
        actual=None,
        suggestion=None,
        identity=None,
        exception_type="GeneratedCardValidationError"):
    """Build a JSON-safe, human-readable validation problem record."""
    return {
        "problem_id": _generated_problem_id(
            code,
            path,
            identity),
        "code": code,
        "title": _bounded_problem_text(title, 200),
        "message": _bounded_problem_text(message, 1_000),
        "location": _bounded_problem_text(location, 500),
        "scope": scope,
        "path": path,
        "card_index": card_index,
        "card_number": (
            card_index + 1
            if card_index is not None
            else None),
        "term": (
            _bounded_problem_text(term, 200)
            if term is not None
            else None),
        "field_name": field_name,
        "expected": _bounded_problem_value(expected),
        "actual": _bounded_problem_value(actual),
        "suggestion": (
            _bounded_problem_text(suggestion, 1_000)
            if suggestion is not None
            else None),
        "overrideable": bool(overrideable),
        "exception_type": exception_type,
    }


def _card_location(card_index, note_data, term_field, field_name=None):
    term = (
        note_data.get(term_field)
        if isinstance(note_data, dict)
        else None)
    if not isinstance(term, str) or not term.strip():
        term = None
    label = f"Card {card_index + 1}"
    if term is not None:
        label += f' (“{term}”)'
    if field_name is not None:
        label += f" → {field_name}"
    return label, term


def _field_path(card_index, field_name):
    return (
        f"$.cards[{card_index}]"
        f"[{json.dumps(field_name, ensure_ascii=False)}]")


def _public_generated_validation_report(report):
    """Remove canonical card data before persisting/displaying a report."""
    return {
        key: value
        for key, value in report.items()
        if key != "canonical_response"
    }


def inspect_generated_response(
        text,
        pipeline=None,
        *,
        optional_fields=(),
        enforce_sentence_count=True,
        require_sentence_translations=True):
    """Describe every detectable generated-card problem without raising.

    Structural problems cannot be manually overridden because accepting them
    could make deck packaging ambiguous or unsafe. Content constraints are
    marked overrideable: callers may record an explicit human decision and
    later package through the corresponding structurally validated path.
    """
    pipeline = pipeline or pipeline_store.default_pipeline()
    pipeline_store.validate_pipelines((pipeline,))
    language_settings = pipeline_store.get_language_settings(
        pipeline,
        pipeline.language_key)
    enabled_cards = pipeline_store.get_enabled_cards(pipeline)
    expected_fields = get_response_field_names(
        pipeline,
        include_sentence_translations=require_sentence_translations)
    expected_field_set = set(expected_fields)
    optional_field_set = set(optional_fields)
    unsupported_optional_fields = (
        optional_field_set - expected_field_set)
    if unsupported_optional_fields:
        raise ValueError(
            "Optional response fields are not enabled by this pipeline: "
            + ", ".join(sorted(unsupported_optional_fields)))
    required_field_set = expected_field_set - optional_field_set
    term_field = pipeline_store.get_language(
        pipeline.language_key).term_field
    source_language = pipeline_store.get_language(pipeline.language_key)
    requires_english_translation = (
        pipeline_store.translation_target_allowed(
            pipeline.language_key,
            "english"))
    empty_allowed_fields = {
        pipeline_store.response_field_name(field_setting)
        for field_setting
        in pipeline_store.get_requested_field_settings(pipeline)
        if field_setting.field_key == "nuance"
    }
    requested_field_settings = (
        pipeline_store.get_requested_field_settings(pipeline))
    translation_fields = {
        field_setting.target_language_key: (
            pipeline_store.response_field_name(field_setting))
        for field_setting in requested_field_settings
        if field_setting.field_key == "translation"
    }
    dictionary_meaning_fields = {
        field_setting.target_language_key: (
            pipeline_store.response_field_name(field_setting))
        for field_setting in requested_field_settings
        if field_setting.field_key == "dictionary_meaning"
    }
    translation_definition_pairs = tuple(
        (
            target_language_key,
            translation_fields[target_language_key],
            dictionary_meaning_fields[target_language_key],
        )
        for target_language_key in sorted(
            set(translation_fields) & set(dictionary_meaning_fields))
    )

    problems = []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        if isinstance(error, json.JSONDecodeError):
            location = (
                f"Response JSON, line {error.lineno}, "
                f"column {error.colno}")
            actual = {
                "line": error.lineno,
                "column": error.colno,
                "character_offset": error.pos,
            }
        else:
            location = "Response JSON"
            actual = {"python_type": type(text).__name__}
        problems.append(_generated_problem(
            "invalid_json",
            "Response is not valid JSON",
            str(error),
            location=location,
            expected="A JSON object containing a cards array.",
            actual=actual,
            suggestion=(
                "Retry the request or repair the JSON before reviewing "
                "individual cards."),
            exception_type=type(error).__name__))
        return {
            "valid": False,
            "syntax_valid": False,
            "structurally_valid": False,
            "can_manually_accept": False,
            "problem_count": 1,
            "overrideable_problem_count": 0,
            "non_overrideable_problem_count": 1,
            "problems": problems,
            "canonical_response": None,
        }

    parser = parsed
    root_structurally_valid = True
    if isinstance(parsed, dict):
        root_keys = set(parsed)
        if root_keys != {"cards"}:
            root_structurally_valid = False
            missing = {"cards"} - root_keys
            unexpected = root_keys - {"cards"}
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unexpected:
                details.append(
                    "unexpected " + ", ".join(sorted(unexpected)))
            problems.append(_generated_problem(
                "invalid_root_fields",
                "Response object has the wrong fields",
                "The generated JSON object must contain only "
                '"cards": ' + "; ".join(details),
                expected=['cards'],
                actual=sorted(root_keys),
                suggestion=(
                    "Retry or edit the response so its root object contains "
                    "only the cards array.")))
        parser = parsed.get("cards")

    parser_is_list = isinstance(parser, list)
    if not parser_is_list:
        problems.append(_generated_problem(
            "cards_not_array",
            "Cards value is not an array",
            "Generated card data must be an array.",
            path="$.cards",
            location="Response → cards",
            expected="array",
            actual=type(parser).__name__,
            suggestion=(
                "Retry or edit the response so cards is a JSON array.")))

    canonical_notes = []
    cards_structurally_valid = parser_is_list
    if parser_is_list:
        for card_index, raw_note_data in enumerate(parser):
            if not isinstance(raw_note_data, dict):
                cards_structurally_valid = False
                problems.append(_generated_problem(
                    "card_not_object",
                    "Card is not an object",
                    "Every generated card must be a JSON object.",
                    path=f"$.cards[{card_index}]",
                    location=f"Card {card_index + 1}",
                    scope="card",
                    card_index=card_index,
                    expected="object",
                    actual=type(raw_note_data).__name__,
                    suggestion=(
                        "Retry or replace this value with an object containing "
                        "the required card fields.")))
                continue

            note_data = dict(raw_note_data)
            canonical_notes.append(note_data)
            location, term = _card_location(
                card_index,
                note_data,
                term_field)
            note_keys = set(note_data)
            missing = required_field_set - note_keys
            unexpected = note_keys - expected_field_set
            if missing or unexpected:
                cards_structurally_valid = False
                details = []
                if missing:
                    details.append(
                        "missing " + ", ".join(sorted(missing)))
                if unexpected:
                    details.append(
                        "unexpected " + ", ".join(sorted(unexpected)))
                problems.append(_generated_problem(
                    "invalid_card_fields",
                    "Card has the wrong fields",
                    "Generated card fields are invalid: "
                    + "; ".join(details),
                    path=f"$.cards[{card_index}]",
                    location=location,
                    scope="card",
                    card_index=card_index,
                    term=term,
                    expected=list(expected_fields),
                    actual=sorted(note_keys),
                    suggestion=(
                        "Retry or edit this card so it contains exactly the "
                        "fields selected in Card Setup.")))

            string_fields = set()
            for field_name in expected_fields:
                if field_name not in note_data:
                    continue
                field = note_data[field_name]
                field_location, field_term = _card_location(
                    card_index,
                    note_data,
                    term_field,
                    field_name)
                if not isinstance(field, str):
                    cards_structurally_valid = False
                    problems.append(_generated_problem(
                        "field_not_string",
                        "Field value is not text",
                        "All generated card fields must be strings.",
                        path=_field_path(card_index, field_name),
                        location=field_location,
                        scope="field",
                        card_index=card_index,
                        term=field_term,
                        field_name=field_name,
                        expected="string",
                        actual=type(field).__name__,
                        suggestion=(
                            "Retry or replace this value with JSON text."),
                        exception_type="TypeError"))
                    continue
                string_fields.add(field_name)
                if not field.strip():
                    if field_name in empty_allowed_fields:
                        note_data[field_name] = ""
                        continue
                    problems.append(_generated_problem(
                        "empty_required_field",
                        "Required field is empty",
                        f'Generated field "{field_name}" cannot be empty.',
                        path=_field_path(card_index, field_name),
                        location=field_location,
                        scope="field",
                        overrideable=True,
                        card_index=card_index,
                        term=field_term,
                        field_name=field_name,
                        expected="Non-blank text",
                        actual="Blank text",
                        suggestion=(
                            "Retry, edit the response, or accept this empty "
                            "field after checking the resulting card.")))
                if field_name == "Sentences":
                    unsupported_tags = [
                        tag
                        for tag in _HTML_TAG_PATTERN.findall(field)
                        if tag.casefold() not in {
                            "<strong>",
                            "</strong>",
                        }
                    ]
                    if unsupported_tags:
                        problems.append(_generated_problem(
                            "sentence_contains_unsupported_html",
                            "Example sentence contains unsupported HTML",
                            "Generated example sentences may contain only "
                            "literal <strong> and </strong> tags around the "
                            "defined usage. Other tags and tag attributes are "
                            "not allowed.",
                            path=_field_path(card_index, field_name),
                            location=field_location,
                            scope="field",
                            overrideable=False,
                            card_index=card_index,
                            term=field_term,
                            field_name=field_name,
                            expected=(
                                "Plain sentence text with only literal "
                                "<strong>...</strong> markup."),
                            actual={
                                "unsupported_tags": unsupported_tags,
                            },
                            suggestion=(
                                "Retry, or remove every other tag and every "
                                "attribute from the strong tags."),
                            identity=field_name))
                elif (
                        field_name != SENTENCE_TRANSLATIONS_FIELD_NAME
                        and _HTML_TAG_PATTERN.search(field)):
                    problems.append(_generated_problem(
                        "generated_field_contains_html",
                        "Generated field contains HTML",
                        "Generated terms, definitions, translations, and "
                        "other card fields must be plain text. HTML here "
                        "would be passed directly into the Anki note.",
                        path=_field_path(card_index, field_name),
                        location=field_location,
                        scope="field",
                        overrideable=False,
                        card_index=card_index,
                        term=field_term,
                        field_name=field_name,
                        expected="Plain text without HTML tags.",
                        actual=field,
                        suggestion=(
                            "Retry or remove the HTML after checking that no "
                            "meaning was lost."),
                        identity=field_name))

            for (
                    target_language_key,
                    translation_field,
                    dictionary_meaning_field,
                    ) in translation_definition_pairs:
                if not {
                        translation_field,
                        dictionary_meaning_field} <= string_fields:
                    continue
                translation_text = note_data[
                    translation_field].strip()
                dictionary_text = note_data[
                    dictionary_meaning_field].strip()
                if (
                        not translation_text
                        or translation_text.casefold()
                        != dictionary_text.casefold()):
                    continue
                target_language = pipeline_store.get_language(
                    target_language_key).name
                problems.append(_generated_problem(
                    "translation_duplicates_dictionary_meaning",
                    "Translation duplicates the dictionary explanation",
                    "A translation must be a concise equivalent, while the "
                    "dictionary meaning must explain the selected sense. "
                    "Returning the same text for both loses the requested "
                    "explanation.",
                    path=f"$.cards[{card_index}]",
                    location=location,
                    scope="card",
                    overrideable=True,
                    card_index=card_index,
                    term=term,
                    expected={
                        translation_field: (
                            f"A concise {target_language} equivalent."),
                        dictionary_meaning_field: (
                            f"A distinct {target_language} explanation."),
                    },
                    actual={
                        translation_field: translation_text,
                        dictionary_meaning_field: dictionary_text,
                    },
                    suggestion=(
                        "Retry, or rewrite the dictionary meaning as a plain "
                        "explanation of the sense."),
                    identity=target_language_key))

            if enforce_sentence_count and "Sentences" in string_fields:
                sentences = [
                    sentence.strip()
                    for sentence in note_data["Sentences"].split("|")
                    if sentence.strip()
                ]
                if len(sentences) != 4:
                    field_location, field_term = _card_location(
                        card_index,
                        note_data,
                        term_field,
                        "Sentences")
                    problems.append(_generated_problem(
                        "wrong_sentence_count",
                        "Example sentence count is wrong",
                        "Each detailed card must contain exactly four "
                        "pipe-separated example sentences.",
                        path=_field_path(card_index, "Sentences"),
                        location=field_location,
                        scope="field",
                        overrideable=True,
                        card_index=card_index,
                        term=field_term,
                        field_name="Sentences",
                        expected=4,
                        actual=len(sentences),
                        suggestion=(
                            "Retry, add/remove pipe-separated sentences, or "
                            "accept this count after reviewing the card.")))

            if (
                    "Sentences" in string_fields
                    and SENTENCE_TRANSLATIONS_FIELD_NAME in string_fields):
                sentence_parts = [
                    sentence.strip()
                    for sentence in note_data["Sentences"].split("|")
                ]
                translation_parts = [
                    translation.strip()
                    for translation
                    in note_data[
                        SENTENCE_TRANSLATIONS_FIELD_NAME].split("|")
                ]
                if len(sentence_parts) != len(translation_parts):
                    field_location, field_term = _card_location(
                        card_index,
                        note_data,
                        term_field,
                        SENTENCE_TRANSLATIONS_FIELD_NAME)
                    problems.append(_generated_problem(
                        "sentence_translation_count_mismatch",
                        "Example translations do not match the sentences",
                        "Each pipe-separated example sentence must have one "
                        "English translation in the same position.",
                        path=_field_path(
                            card_index,
                            SENTENCE_TRANSLATIONS_FIELD_NAME),
                        location=field_location,
                        scope="field",
                        overrideable=True,
                        card_index=card_index,
                        term=field_term,
                        field_name=SENTENCE_TRANSLATIONS_FIELD_NAME,
                        expected={
                            "translation_count": len(sentence_parts),
                            "alignment": (
                                "One translation for each sentence, in the "
                                "same order."),
                        },
                        actual={
                            "translation_count": len(translation_parts),
                        },
                        suggestion=(
                            "Retry, or add/remove pipe-separated English "
                            "translations so their positions match the "
                            "example sentences.")))
                blank_translations = [
                    index + 1
                    for index, translation in enumerate(translation_parts)
                    if not translation
                ]
                if blank_translations:
                    field_location, field_term = _card_location(
                        card_index,
                        note_data,
                        term_field,
                        SENTENCE_TRANSLATIONS_FIELD_NAME)
                    problems.append(_generated_problem(
                        "blank_sentence_translation",
                        "An example translation is blank",
                        "Every example sentence needs a non-blank English "
                        "translation in the same position.",
                        path=_field_path(
                            card_index,
                            SENTENCE_TRANSLATIONS_FIELD_NAME),
                        location=field_location,
                        scope="field",
                        overrideable=True,
                        card_index=card_index,
                        term=field_term,
                        field_name=SENTENCE_TRANSLATIONS_FIELD_NAME,
                        expected=(
                            "Non-blank translations for every example "
                            "sentence."),
                        actual={
                            "blank_translation_positions": (
                                blank_translations),
                        },
                        suggestion=(
                            "Retry, or fill each listed translation before "
                            "packaging.")))
                markup_translations = [
                    index + 1
                    for index, translation in enumerate(translation_parts)
                    if _HTML_TAG_PATTERN.search(translation)
                ]
                if markup_translations:
                    field_location, field_term = _card_location(
                        card_index,
                        note_data,
                        SENTENCE_TRANSLATIONS_FIELD_NAME)
                    problems.append(_generated_problem(
                        "sentence_translation_contains_html",
                        "An example translation contains HTML",
                        "English example translations must be plain text; "
                        "strong tags and other HTML belong only in the "
                        "source-language sentence.",
                        path=_field_path(
                            card_index,
                            SENTENCE_TRANSLATIONS_FIELD_NAME),
                        location=field_location,
                        scope="field",
                        overrideable=False,
                        card_index=card_index,
                        term=field_term,
                        field_name=SENTENCE_TRANSLATIONS_FIELD_NAME,
                        expected="Plain-text translations without HTML.",
                        actual={
                            "translation_positions_with_html": (
                                markup_translations),
                        },
                        suggestion=(
                            "Retry, or remove the HTML after checking each "
                            "translation.")))

                if requires_english_translation:
                    for item_index, translation in enumerate(
                            translation_parts):
                        if not translation:
                            continue
                        source_sentence = (
                            sentence_parts[item_index]
                            if item_index < len(sentence_parts)
                            else "")
                        reason, plain_source = (
                            _sentence_translation_language_issue(
                                source_sentence,
                                translation,
                                source_language.model_language_key))
                        if reason is None:
                            continue
                        item_number = item_index + 1
                        item_path = (
                            _field_path(
                                card_index,
                                SENTENCE_TRANSLATIONS_FIELD_NAME)
                            + f"[{item_index}]")
                        if reason == "exact_source_copy":
                            message = (
                                "This English translation is an exact copy "
                                "of its source-language sentence after "
                                "strong emphasis markup is removed.")
                        else:
                            message = (
                                "This field is required to be English, but "
                                f"item {item_number} contains "
                                f"{source_language.name} script and no Latin "
                                "alphabetic text.")
                        field_location, field_term = _card_location(
                            card_index,
                            note_data,
                            term_field,
                            SENTENCE_TRANSLATIONS_FIELD_NAME)
                        problems.append(_generated_problem(
                            "sentence_translation_not_english",
                            "Example translation is not English",
                            message,
                            path=item_path,
                            location=(
                                f"{field_location} → item {item_number}"),
                            scope="field",
                            overrideable=False,
                            card_index=card_index,
                            term=field_term,
                            field_name=SENTENCE_TRANSLATIONS_FIELD_NAME,
                            expected=(
                                "A plain English translation of source item "
                                f"{item_number}."),
                            actual={
                                "translation": translation,
                                "source_sentence": plain_source,
                                "reason": reason,
                            },
                            suggestion=(
                                "Retry or replace this item with an English "
                                "translation of the aligned source sentence."),
                            identity={
                                "field": SENTENCE_TRANSLATIONS_FIELD_NAME,
                                "item_index": item_index,
                                "reason": reason,
                            }))

            if "Sentences" in string_fields:
                sentences = [
                    sentence.strip()
                    for sentence in note_data["Sentences"].split("|")
                    if sentence.strip()
                ]
                missing_emphasis = [
                    index + 1
                    for index, sentence in enumerate(sentences)
                    if not _sentence_has_emphasized_usage(sentence)
                ]
                if missing_emphasis:
                    field_location, field_term = _card_location(
                        card_index,
                        note_data,
                        term_field,
                        "Sentences")
                    problems.append(_generated_problem(
                        "missing_sentence_emphasis",
                        "Defined usage is not emboldened",
                        "Every generated example sentence must use "
                        "<strong>...</strong> around every occurrence that "
                        "expresses this card's particular meaning.",
                        path=_field_path(card_index, "Sentences"),
                        location=field_location,
                        scope="field",
                        overrideable=True,
                        card_index=card_index,
                        term=field_term,
                        field_name="Sentences",
                        expected=(
                            "At least one nonempty <strong>...</strong> "
                            "usage in every example sentence."),
                        actual={
                            "sentences_without_emphasis": missing_emphasis,
                        },
                        suggestion=(
                            "Retry, or add strong tags around the intended "
                            "usage in each listed sentence.")))

                if pipeline.language_key.startswith("classical_chinese"):
                    expected_term = note_data.get(term_field)
                    unexpected_spans = []
                    if isinstance(expected_term, str):
                        for sentence_index, sentence in enumerate(
                                sentences,
                                start=1):
                            for span in _STRONG_SPAN_PATTERN.findall(sentence):
                                emphasized = html.unescape(
                                    _STRONG_TAG_PATTERN.sub("", span))
                                if emphasized == expected_term:
                                    continue
                                unexpected_spans.append({
                                    "sentence": sentence_index,
                                    "emphasized": emphasized,
                                })
                    if unexpected_spans:
                        field_location, field_term = _card_location(
                            card_index,
                            note_data,
                            term_field,
                            "Sentences")
                        problems.append(_generated_problem(
                            "unexpected_emphasized_form",
                            "The wrong expression is emboldened",
                            "Classical Chinese strong tags must enclose "
                            "exactly the card's complete term. They cannot "
                            "include a neighbouring character or mark a "
                            "different expression.",
                            path=_field_path(card_index, "Sentences"),
                            location=field_location,
                            scope="field",
                            overrideable=True,
                            card_index=card_index,
                            term=field_term,
                            field_name="Sentences",
                            expected=expected_term,
                            actual=unexpected_spans,
                            suggestion=(
                                "Retry, or move each strong tag so it encloses "
                                "only the complete card term.")))

            if required_field_set <= string_fields:
                for card in enabled_cards:
                    effective_fields = pipeline_store.get_effective_fields(
                        language_settings,
                        card)
                    if any(
                            note_data[
                                pipeline_store.response_field_name(
                                    field_setting)].strip()
                            for field_setting in effective_fields):
                        continue
                    direction_name = pipeline_store.get_direction(
                        card.direction_key).name
                    problems.append(_generated_problem(
                        "empty_card_direction",
                        "Card direction has no meaning content",
                        f'"{direction_name}" has no non-empty meaning fields.',
                        path=f"$.cards[{card_index}]",
                        location=location,
                        scope="card",
                        overrideable=True,
                        card_index=card_index,
                        term=term,
                        expected=(
                            "At least one non-empty meaning field for "
                            f"{direction_name}."),
                        actual="All effective meaning fields are blank.",
                        suggestion=(
                            "Retry, fill one meaning field, disable this card "
                            "direction, or accept the empty direction."),
                        identity=card.direction_key))

    structurally_valid = (
        root_structurally_valid
        and parser_is_list
        and cards_structurally_valid)
    canonical_response = (
        {"cards": canonical_notes}
        if structurally_valid
        else None)
    non_overrideable_count = sum(
        not problem["overrideable"]
        for problem in problems)
    overrideable_count = len(problems) - non_overrideable_count
    return {
        "valid": not problems,
        "syntax_valid": True,
        "structurally_valid": structurally_valid,
        "can_manually_accept": (
            structurally_valid
            and non_overrideable_count == 0),
        "problem_count": len(problems),
        "overrideable_problem_count": overrideable_count,
        "non_overrideable_problem_count": non_overrideable_count,
        "problems": problems,
        "canonical_response": canonical_response,
    }


def get_api_key():
    environment_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    return credential_store.load_api_key()


def get_response_field_names(
        pipeline,
        *,
        include_sentence_translations=True):
    pipeline_store.validate_pipelines((pipeline,))
    language = pipeline_store.get_language(
        pipeline.language_key)
    field_names = [language.term_field]
    if pipeline_store.requires_sentences(pipeline):
        field_names.append("Sentences")
        if include_sentence_translations:
            field_names.append(SENTENCE_TRANSLATIONS_FIELD_NAME)
    field_names.extend(
        pipeline_store.response_field_name(field_setting)
        for field_setting
        in pipeline_store.get_requested_field_settings(pipeline))
    return tuple(field_names)


def build_response_format(
        pipeline,
        *,
        optional_fields=(),
        require_sentence_translations=True,
        sentence_collections_as_arrays=False,
        include_source_context_translations=False,
        split_source_context_cards=False):
    """Build a strict schema containing only fields selected in the UI."""
    field_names = get_response_field_names(
        pipeline,
        include_sentence_translations=require_sentence_translations)
    optional_field_set = set(optional_fields)
    unsupported_optional_fields = (
        optional_field_set - set(field_names))
    if unsupported_optional_fields:
        raise ValueError(
            "Optional response fields are not enabled by this pipeline: "
            + ", ".join(sorted(unsupported_optional_fields)))
    if (
            include_source_context_translations
            and (
                not sentence_collections_as_arrays
                or not require_sentence_translations
                or "Sentences" not in optional_field_set)):
        raise ValueError(
            "Source-context translation maps require the array response "
            "format with optional contextual Sentences.")
    if (
            split_source_context_cards
            and not include_source_context_translations):
        raise ValueError(
            "Split source-context cards require a source-context "
            "translation map.")
    detail_name = (
        "context"
        if pipeline_store.requires_sentences(pipeline)
        else "simple")
    if optional_field_set:
        detail_name += "_optional_" + "_".join(
            field_name.lower().replace(" ", "_")
            for field_name in sorted(optional_field_set))
    if sentence_collections_as_arrays:
        detail_name += "_arrays"
    if include_source_context_translations:
        detail_name += "_source_context_map"
    if split_source_context_cards:
        detail_name += "_split_cards"

    def collection_schema():
        return {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
        }

    def card_schema_for(names):
        return {
            "type": "object",
            "properties": {
                field_name: (
                    collection_schema()
                    if (
                        sentence_collections_as_arrays
                        and field_name in {
                            "Sentences",
                            SENTENCE_TRANSLATIONS_FIELD_NAME,
                        })
                    else {"type": "string"})
                for field_name in names
            },
            "required": list(names),
            "additionalProperties": False,
        }

    card_schema = card_schema_for(field_names)
    if optional_field_set and not split_source_context_cards:
        # Structured Outputs requires every property in an object branch to
        # be required. Represent genuine omission as two strict object shapes
        # rather than declaring an optional property in one shape.
        if len(optional_field_set) != 1:
            raise ValueError(
                "Only one optional response field is currently supported.")
        without_optional = tuple(
            field_name
            for field_name in field_names
            if (
                field_name not in optional_field_set
                and (
                    not include_source_context_translations
                    or field_name != SENTENCE_TRANSLATIONS_FIELD_NAME)))
        card_schema = {
            "anyOf": [
                card_schema_for(without_optional),
                card_schema_for(field_names),
            ],
        }
    if split_source_context_cards:
        contextual_field_names = tuple(
            field_name
            for field_name in field_names
            if field_name not in {
                "Sentences",
                SENTENCE_TRANSLATIONS_FIELD_NAME,
            })
        root_properties = {
            SOURCE_CONTEXTUAL_CARDS_KEY: {
                "type": "array",
                "items": card_schema_for(contextual_field_names),
            },
            SOURCE_ADDITIONAL_SENSE_CARDS_KEY: {
                "type": "array",
                "items": card_schema_for(field_names),
            },
        }
        root_required = [
            SOURCE_CONTEXTUAL_CARDS_KEY,
            SOURCE_ADDITIONAL_SENSE_CARDS_KEY,
        ]
    else:
        root_properties = {
            "cards": {
                "type": "array",
                "items": card_schema,
            },
        }
        root_required = ["cards"]
    if include_source_context_translations:
        root_properties[SOURCE_CONTEXT_TRANSLATIONS_KEY] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    SOURCE_CONTEXT_ID_FIELD_NAME: {
                        "type": "string",
                    },
                    SOURCE_CONTEXT_TRANSLATION_FIELD_NAME: {
                        "type": "string",
                    },
                },
                "required": [
                    SOURCE_CONTEXT_ID_FIELD_NAME,
                    SOURCE_CONTEXT_TRANSLATION_FIELD_NAME,
                ],
                "additionalProperties": False,
            },
        }
        root_required.append(SOURCE_CONTEXT_TRANSLATIONS_KEY)
    return {
        "type": "json_schema",
        "name": bounded_response_format_name(
            f"autoanki_{pipeline.language_key}_{detail_name}_cards"),
        "strict": True,
        "schema": {
            "type": "object",
            "properties": root_properties,
            "required": root_required,
            "additionalProperties": False,
        },
    }


def grouped_contextual_field_constants(pipeline, word, contexts_by_id):
    """Return immutable lexical facts for an exactly audited occurrence.

    Most contextual senses remain model-selected. A small source-specific
    constraint is appropriate only where the retained syntax is unequivocal
    and repeated live audits have shown that prompt guidance alone is not a
    sufficient guard.
    """
    if (
            pipeline.language_key != "classical_chinese_wang_bi"
            or getattr(word, "surface", None) != "始"
            or not isinstance(contexts_by_id, dict)):
        return {}
    context = contexts_by_id.get(getattr(word, "context_id", None))
    context_text = getattr(context, "text", None)
    context_start = getattr(context, "start_offset", None)
    word_start = getattr(word, "start_offset", None)
    word_end = getattr(word, "end_offset", None)
    if (
            not isinstance(context_text, str)
            or isinstance(context_start, bool)
            or not isinstance(context_start, int)
            or isinstance(word_start, bool)
            or not isinstance(word_start, int)
            or isinstance(word_end, bool)
            or not isinstance(word_end, int)):
        return {}
    relative_start = word_start - context_start
    relative_end = word_end - context_start
    if (
            relative_start < 0
            or relative_end > len(context_text)
            or relative_start >= relative_end
            or context_text[relative_start:relative_end] != "始"
            or not context_text[
                max(0, relative_start - len("天地之")):relative_start
            ].endswith("天地之")):
        return {}
    requested_fields = set(get_response_field_names(
        pipeline,
        include_sentence_translations=True))
    return {
        field_name: value
        for field_name, value in
        _WANG_BI_TIANDI_ZHI_SHI_CONTEXTUAL_FIELDS.items()
        if field_name in requested_fields
    }


def compact_contextual_field_constants(pipeline, word, contexts_by_id):
    """Return only concise immutable facts suitable for compact v9 checks."""
    constants = grouped_contextual_field_constants(
        pipeline,
        word,
        contexts_by_id)
    if (
            pipeline.language_key == "classical_chinese_wang_bi"
            and isinstance(contexts_by_id, dict)):
        context = contexts_by_id.get(getattr(word, "context_id", None))
        audited = _WANG_BI_COMPACT_CONTEXTUAL_FIELDS.get((
            getattr(context, "text", None),
            getattr(word, "surface", None),
        ))
        if audited is not None:
            requested_fields = set(get_response_field_names(
                pipeline,
                include_sentence_translations=True))
            constants = {
                **constants,
                **{
                    field_name: value
                    for field_name, value in audited.items()
                    if field_name in requested_fields
                },
            }
    immutable_field_names = {
        "Translation (English)",
        "Pronunciation (English)",
        "Part of Speech (English)",
    }
    return {
        field_name: value
        for field_name, value in constants.items()
        if field_name in immutable_field_names
    }


def build_grouped_source_response_format(pipeline, chunk):
    """Build an exact rank-keyed schema for one retained-source chunk.

    The rank and context property names are taken from the immutable chunk,
    which makes missing, duplicate, and unrequested results structurally
    impossible in a successful Structured Outputs response.
    """
    pipeline_store.validate_pipelines((pipeline,))
    if not pipeline_store.requires_sentences(pipeline):
        raise ValueError(
            "Grouped source results require the Context card direction.")
    try:
        words = tuple(chunk.words)
        contexts = tuple(chunk.contexts)
        chunk_id = chunk.chunk_id
    except AttributeError as error:
        raise TypeError(
            "A grouped source response format requires a generation chunk."
        ) from error
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("A source generation chunk requires an identifier.")
    if not words:
        raise ValueError(
            "A grouped source response format requires at least one word.")

    rank_keys = []
    rank_words = {}
    word_context_ids = []
    for word in words:
        rank = getattr(word, "rank", None)
        if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank < 1):
            raise ValueError(
                "Every grouped source word requires a positive integer rank.")
        rank_key = str(rank)
        if rank_key in rank_keys:
            raise ValueError(
                f"Grouped source word rank {rank} appears more than once.")
        surface = getattr(word, "surface", None)
        if not isinstance(surface, str) or not surface:
            raise ValueError(
                "Every grouped source word requires a non-empty surface.")
        rank_keys.append(rank_key)
        rank_words[rank_key] = word
        word_context_ids.append(getattr(word, "context_id", None))
    rank_keys.sort(key=int)

    context_ids = []
    contexts_by_id = {}
    for context in contexts:
        context_id = getattr(context, "context_id", None)
        if not isinstance(context_id, str) or not context_id:
            raise ValueError(
                "Every grouped source context requires a non-empty ID.")
        if context_id in context_ids:
            raise ValueError(
                f"Grouped source context {context_id!r} appears more than "
                "once.")
        context_ids.append(context_id)
        contexts_by_id[context_id] = context
    context_id_set = set(context_ids)
    if any(
            not isinstance(context_id, str)
            or context_id not in context_id_set
            for context_id in word_context_ids):
        raise ValueError(
            "Every grouped source word must reference a context in its "
            "chunk.")

    term_field = pipeline_store.get_language(
        pipeline.language_key).term_field
    lexical_field_names = tuple(
        field_name
        for field_name in get_response_field_names(
            pipeline,
            include_sentence_translations=True)
        if field_name not in {
            term_field,
            "Sentences",
            SENTENCE_TRANSLATIONS_FIELD_NAME,
        })

    def string_properties(field_names, surface, role):
        return {
            field_name: {
                "type": "string",
                "description": (
                    f"{field_name} for only the {role} of the exact complete "
                    f"requested term {surface!r}; never describe or "
                    "pronounce only one component character or substring."),
            }
            for field_name in field_names
        }

    def contextual_sense_schema(word):
        surface = word.surface
        properties = string_properties(
            lexical_field_names,
            surface,
            "contextual sense")
        for field_name, required_value in (
                grouped_contextual_field_constants(
                    pipeline,
                    word,
                    contexts_by_id).items()):
            properties[field_name]["enum"] = [required_value]
            properties[field_name]["description"] += (
                f" For this exact audited occurrence, return exactly "
                f"{required_value!r}.")
        return {
            "type": "object",
            "description": (
                "Lexical fields for only the sense and grammatical role of "
                f"the exact marked occurrence of {surface!r}."),
            "properties": properties,
            "required": list(lexical_field_names),
            "additionalProperties": False,
        }

    generated_field_names = (
        *lexical_field_names,
        "Sentences",
        SENTENCE_TRANSLATIONS_FIELD_NAME,
    )

    def additional_sense_schema(surface):
        properties = string_properties(
            lexical_field_names,
            surface,
            "additional sense")
        sentence_item_schema = {
            "type": "string",
            "description": (
                "One complete natural source-language example containing "
                f"the exact complete requested term {surface!r} exactly "
                f"once, as literal <strong>{surface}</strong>. Do not use "
                "a second occurrence anywhere, including inside a "
                "compound. Never mark only a component or substring, and "
                "never use em or any other HTML tag."),
        }
        if (
                pipeline.language_key in _UNBOUNDED_TERM_LANGUAGES
                and len(surface) == 1):
            escaped_surface = re.escape(surface)
            sentence_item_schema["pattern"] = (
                f"^[^{escaped_surface}]*"
                f"<strong>{escaped_surface}</strong>"
                f"[^{escaped_surface}]*$")
        properties["Sentences"] = {
            "type": "array",
            "items": sentence_item_schema,
            "minItems": 4,
            "maxItems": 4,
        }
        properties[SENTENCE_TRANSLATIONS_FIELD_NAME] = {
            "type": "array",
            "items": {
                "type": "string",
                "description": (
                    "One complete natural English translation of the "
                    "source sentence at the same array position; never copy "
                    "source-language text and never include HTML."),
            },
            "minItems": 4,
            "maxItems": 4,
        }
        return {
            "type": "object",
            "description": (
                "One genuinely disjoint lexical sense of the exact complete "
                f"requested term {surface!r}; never a sense of one component "
                "character or substring."),
            "properties": properties,
            "required": list(generated_field_names),
            "additionalProperties": False,
        }

    def term_result_schema(word):
        surface = word.surface
        return {
            "type": "object",
            "properties": {
                SOURCE_CONTEXTUAL_SENSE_KEY: contextual_sense_schema(
                    word),
                SOURCE_ADDITIONAL_SENSES_KEY: {
                    "type": "array",
                    "description": (
                        "Only genuinely disjoint common senses of the exact "
                        f"complete requested term {surface!r}. Use an empty "
                        "array rather than defining a component or repeating "
                        "the contextual sense."),
                    "items": additional_sense_schema(surface),
                },
            },
            "required": [
                SOURCE_CONTEXTUAL_SENSE_KEY,
                SOURCE_ADDITIONAL_SENSES_KEY,
            ],
            "additionalProperties": False,
        }

    root_schema = {
        "type": "object",
        "properties": {
            SOURCE_TERM_RESULTS_KEY: {
                "type": "object",
                "properties": {
                    rank_key: term_result_schema(rank_words[rank_key])
                    for rank_key in rank_keys
                },
                "required": rank_keys,
                "additionalProperties": False,
            },
            SOURCE_CONTEXT_TRANSLATIONS_KEY: {
                "type": "object",
                "properties": {
                    context_id: {
                        "type": "string",
                        "description": (
                            "One complete natural English translation of "
                            "this entire retained source context, without "
                            "HTML or a pipe delimiter."),
                    }
                    for context_id in context_ids
                },
                "required": context_ids,
                "additionalProperties": False,
            },
        },
        "required": [
            SOURCE_TERM_RESULTS_KEY,
            SOURCE_CONTEXT_TRANSLATIONS_KEY,
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "name": bounded_response_format_name(
            f"autoanki_{pipeline.language_key}_{chunk_id}_grouped_source"),
        "strict": True,
        "schema": root_schema,
    }


def build_compact_source_response_format(pipeline):
    """Build the fixed, rank-addressed v9 retained-source schema.

    Unlike v8's per-chunk schema, this shape contains no source ranks, terms,
    or context IDs. One frozen format can therefore serve every chunk created
    from the same generation pipeline; exact membership remains recoverable
    from the integer rank and context_id fields.
    """
    pipeline_store.validate_pipelines((pipeline,))
    if not pipeline_store.requires_sentences(pipeline):
        raise ValueError(
            "Compact source results require the Context card direction.")

    term_field = pipeline_store.get_language(
        pipeline.language_key).term_field
    lexical_field_names = tuple(
        field_name
        for field_name in get_response_field_names(
            pipeline,
            include_sentence_translations=True)
        if field_name not in {
            term_field,
            "Sentences",
            SENTENCE_TRANSLATIONS_FIELD_NAME,
        })

    def lexical_properties():
        return {
            field_name: {"type": "string"}
            for field_name in lexical_field_names
        }

    contextual_sense_schema = {
        "type": "object",
        "properties": lexical_properties(),
        "required": list(lexical_field_names),
        "additionalProperties": False,
    }
    additional_properties = lexical_properties()
    for field_name in (
            "Sentences",
            SENTENCE_TRANSLATIONS_FIELD_NAME):
        additional_properties[field_name] = {
            "type": "array",
            "items": {
                "type": "string",
                **(
                    {
                        "pattern": (
                            r"^.*<strong>.+</strong>.*$"),
                    }
                    if field_name == "Sentences"
                    else {}),
            },
            "minItems": 4,
            "maxItems": 4,
        }
    additional_sense_schema = {
        "type": "object",
        "properties": additional_properties,
        "required": [
            *lexical_field_names,
            "Sentences",
            SENTENCE_TRANSLATIONS_FIELD_NAME,
        ],
        "additionalProperties": False,
    }
    term_result_schema = {
        "type": "object",
        "properties": {
            SOURCE_RANK_FIELD_NAME: {"type": "integer"},
            SOURCE_CONTEXTUAL_SENSE_KEY: contextual_sense_schema,
            SOURCE_ADDITIONAL_SENSES_KEY: {
                "type": "array",
                "items": additional_sense_schema,
            },
        },
        "required": [
            SOURCE_RANK_FIELD_NAME,
            SOURCE_CONTEXTUAL_SENSE_KEY,
            SOURCE_ADDITIONAL_SENSES_KEY,
        ],
        "additionalProperties": False,
    }
    context_translation_schema = {
        "type": "object",
        "properties": {
            SOURCE_CONTEXT_ID_FIELD_NAME: {"type": "string"},
            SOURCE_CONTEXT_TRANSLATION_FIELD_NAME: {"type": "string"},
        },
        "required": [
            SOURCE_CONTEXT_ID_FIELD_NAME,
            SOURCE_CONTEXT_TRANSLATION_FIELD_NAME,
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "name": bounded_response_format_name(
            f"autoanki_{pipeline.language_key}_compact_source_v9"),
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                SOURCE_TERM_RESULTS_KEY: {
                    "type": "array",
                    "items": term_result_schema,
                },
                SOURCE_CONTEXT_TRANSLATIONS_KEY: {
                    "type": "array",
                    "items": context_translation_schema,
                },
            },
            "required": [
                SOURCE_TERM_RESULTS_KEY,
                SOURCE_CONTEXT_TRANSLATIONS_KEY,
            ],
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

    default_pipeline = None
    if prompt_text is None:
        if prompt_path is None:
            default_pipeline = pipeline_store.default_pipeline()
            prompt_text = prompt_builder.build_prompt(default_pipeline)
        else:
            prompt_text = Path(prompt_path).read_text(encoding="utf-8")
    if response_format is None:
        if default_pipeline is None:
            default_pipeline = pipeline_store.default_pipeline()
        response_format = build_response_format(
            default_pipeline)

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


def validate_generated_cards(
        text,
        pipeline=None,
        *,
        allow_accepted_content_problems=False,
        enforce_sentence_count=True,
        require_sentence_translations=True):
    """Parse and validate generated data without writing an Anki package."""
    report = inspect_generated_response(
        text,
        pipeline,
        enforce_sentence_count=enforce_sentence_count,
        require_sentence_translations=require_sentence_translations)
    remaining_problems = [
        problem
        for problem in report["problems"]
        if (
            not allow_accepted_content_problems
            or not problem["overrideable"])
    ]
    if remaining_problems:
        first = remaining_problems[0]
        exception_class = (
            TypeError
            if first["exception_type"] == "TypeError"
            else GeneratedCardValidationError)
        error = exception_class(first["message"])
        error.validation_report = _public_generated_validation_report(report)
        raise error
    canonical = report["canonical_response"]
    if canonical is None:
        # Defensive guard: every structural problem is non-overrideable, so
        # the branch above should already have raised.
        raise GeneratedCardValidationError(
            "Generated card data is not structurally packageable.")
    return canonical["cards"]


def validate_generated_response(
        text,
        pipeline=None,
        *,
        allow_accepted_content_problems=False,
        enforce_sentence_count=True,
        require_sentence_translations=True):
    """Return the canonical response object after structural validation."""
    return {
        "cards": validate_generated_cards(
            text,
            pipeline,
            allow_accepted_content_problems=(
                allow_accepted_content_problems),
            enforce_sentence_count=enforce_sentence_count,
            require_sentence_translations=require_sentence_translations),
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
        allow_accepted_content_problems=False,
        enforce_sentence_count=True,
        require_sentence_translations=True,
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
    validated_notes = validate_generated_cards(
        text,
        pipeline,
        allow_accepted_content_problems=(
            allow_accepted_content_problems),
        enforce_sentence_count=enforce_sentence_count,
        require_sentence_translations=require_sentence_translations)
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
        sentences = note_data.get("Sentences", "")
        if sentences:
            sentences = sanitize_emphasized_sentences(sentences)
        sentence_translations = note_data.get(
            SENTENCE_TRANSLATIONS_FIELD_NAME,
            "")
        if sentence_translations:
            sentence_translations = sanitize_sentence_translations(
                sentence_translations)
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
                sentences,
                *(
                    note_data.get(
                        response_fields_by_key.get(field_key, ""),
                        "")
                    for field_key, _field_name
                    in templates.CONTENT_FIELDS
                ),
                sentence_translations,
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
    if prompt_text is None and prompt_path is None:
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
