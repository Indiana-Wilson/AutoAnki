"""Detailed, reviewable validators for source-generation responses."""

from collections import Counter
import html
import json
import re

from corpus_pipeline.processing import normalize_word
import pipeline_store
import process_text
from source_generation.local_repair import repair_compact_response


_SENTENCE_TRANSLATIONS_FIELD = (
    process_text.SENTENCE_TRANSLATIONS_FIELD_NAME)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_GROUPED_TERM_RESULTS_KEY = process_text.SOURCE_TERM_RESULTS_KEY
_GROUPED_CONTEXTUAL_SENSE_KEY = (
    process_text.SOURCE_CONTEXTUAL_SENSE_KEY)
_GROUPED_ADDITIONAL_SENSES_KEY = (
    process_text.SOURCE_ADDITIONAL_SENSES_KEY)
_COMPACT_RANK_FIELD = process_text.SOURCE_RANK_FIELD_NAME
_GROUPED_CONTEXT_TRANSLATION_PLACEHOLDER = (
    "[AutoAnki validated source-context translation]")


class ValidatedPipelineResponse(dict):
    """Canonical cards plus non-JSON metadata for durable local-repair audit."""

    def __init__(
            self,
            value,
            *,
            local_repair_audit=None,
            effective_raw_text=None):
        super().__init__(value)
        self.local_repair_audit = local_repair_audit
        self.effective_raw_text = effective_raw_text


def _source_context_translation_map(parsed, chunk):
    """Detach and validate a context-id-to-translation collection."""
    problems = []
    translations = {}
    expected_ids = {
        context.context_id
        for context in chunk.contexts
    }
    missing_marker = object()
    raw_entries = (
        parsed.pop(
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY,
            missing_marker)
        if isinstance(parsed, dict)
        else missing_marker)
    path = f"$.{process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY}"
    if not isinstance(raw_entries, list):
        problems.append(process_text._generated_problem(
            "invalid_source_context_translation_map",
            "Source-context translations are missing or malformed",
            "The response must contain a root source_context_translations "
            "array so each retained context can be matched unambiguously.",
            path=path,
            location="Response → source_context_translations",
            expected="An array containing one object per requested context.",
            actual=(
                "Field omitted"
                if raw_entries is missing_marker
                else type(raw_entries).__name__),
            suggestion=(
                "Retry with the frozen response schema; do not move context "
                "translations into individual cards.")))
        return translations, problems

    seen_ids = set()
    expected_keys = {
        process_text.SOURCE_CONTEXT_ID_FIELD_NAME,
        process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME,
    }
    for entry_index, entry in enumerate(raw_entries):
        entry_path = f"{path}[{entry_index}]"
        if (
                not isinstance(entry, dict)
                or set(entry) != expected_keys
                or not isinstance(
                    entry.get(process_text.SOURCE_CONTEXT_ID_FIELD_NAME),
                    str)
                or not isinstance(
                    entry.get(
                        process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME),
                    str)):
            problems.append(process_text._generated_problem(
                "invalid_source_context_translation_entry",
                "Source-context translation entry is malformed",
                "Every source-context translation must contain exactly the "
                "text fields context_id and translation.",
                path=entry_path,
                location=(
                    "Response → source_context_translations "
                    f"→ item {entry_index + 1}"),
                expected={
                    "context_id": "string",
                    "translation": "string",
                },
                actual=entry,
                suggestion=(
                    "Retry or replace this item with the exact two-field "
                    "object required by the response schema.")))
            continue
        context_id = entry[process_text.SOURCE_CONTEXT_ID_FIELD_NAME]
        translation = entry[
            process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME]
        if context_id in seen_ids:
            problems.append(process_text._generated_problem(
                "duplicate_source_context_translation",
                "Source context was translated more than once",
                "Each requested context_id must occur exactly once in "
                "source_context_translations.",
                path=entry_path,
                location=(
                    "Response → source_context_translations "
                    f"→ item {entry_index + 1}"),
                expected="A context_id not used by an earlier item.",
                actual=context_id,
                suggestion=(
                    "Retry or remove the duplicate context translation."),
                identity=context_id))
            continue
        seen_ids.add(context_id)
        if context_id not in expected_ids:
            problems.append(process_text._generated_problem(
                "unexpected_source_context_translation",
                "An unrequested source context was translated",
                "source_context_translations contains a context_id that is "
                "not present in this source chunk.",
                path=entry_path,
                location=(
                    "Response → source_context_translations "
                    f"→ item {entry_index + 1}"),
                expected=sorted(expected_ids),
                actual=context_id,
                suggestion=(
                    "Retry or remove the unrequested translation."),
                identity=context_id))
            continue
        translations[context_id] = translation

    for context_id in sorted(expected_ids - set(translations)):
        problems.append(process_text._generated_problem(
            "missing_source_context_translation",
            "A retained source context has no translation",
            "Every context_id supplied in this source chunk must have exactly "
            "one complete English translation.",
            path=path,
            location="Response → source_context_translations",
            expected=context_id,
            actual="No matching translation object.",
            suggestion=(
                "Retry and translate every requested context exactly once."),
            identity=context_id))
    return translations, problems


def _normalize_grouped_source_results(
        parsed,
        pipeline,
        chunk,
        *,
        include_source_context_nuance=False):
    """Normalize the rank-keyed v8 response into canonical card candidates.

    V8 deliberately omits the term from every sense object.  The immutable
    request chunk is authoritative: rank selects the exact ``GenerationWord``
    and this function injects its surface before ordinary card validation.
    """
    problems = []
    cards = []
    card_raw_paths = []
    contextual_indices = set()
    additional_indices = set()
    translations = {}
    nuances = {}
    translation_key = process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY
    expected_root_keys = {
        _GROUPED_TERM_RESULTS_KEY,
        translation_key,
    }
    missing_marker = object()

    if not isinstance(parsed, dict):
        problems.append(process_text._generated_problem(
            "invalid_grouped_source_response_root",
            "Grouped source response root is malformed",
            "A grouped source response must be one JSON object containing "
            "only term_results and source_context_translations.",
            path="$",
            location="Response",
            expected=sorted(expected_root_keys),
            actual=type(parsed).__name__,
            suggestion=(
                "Retry using the frozen grouped structured-output schema.")))
        return (
            cards,
            card_raw_paths,
            translations,
            nuances,
            contextual_indices,
            additional_indices,
            problems,
        )

    root_keys = set(parsed)
    if root_keys != expected_root_keys:
        problems.append(process_text._generated_problem(
            "invalid_grouped_source_response_root",
            "Grouped source response root has the wrong fields",
            "A grouped source response must contain exactly term_results and "
            "source_context_translations.",
            path="$",
            location="Response",
            expected=sorted(expected_root_keys),
            actual=sorted(root_keys),
            suggestion=(
                "Retry using the frozen grouped structured-output schema.")))

    term_results = parsed.get(
        _GROUPED_TERM_RESULTS_KEY,
        missing_marker)
    words_by_rank = {}
    for word in chunk.words:
        rank_key = str(word.rank)
        if rank_key in words_by_rank:
            problems.append(process_text._generated_problem(
                "duplicate_requested_source_rank",
                "The source chunk repeats a requested rank",
                "Every grouped result must be addressable by one unique "
                "source rank.",
                path=f"request.words[rank={word.rank}]",
                location=f'Requested term “{word.surface}”',
                scope="request",
                term=word.surface,
                expected="A rank unique within this source chunk.",
                actual=word.rank,
                suggestion=(
                    "Create a new source job from a valid audited plan."),
                identity=word.rank))
            continue
        words_by_rank[rank_key] = word
    expected_rank_keys = set(words_by_rank)
    contexts_by_id = {
        context.context_id: context
        for context in chunk.contexts
    }

    if not isinstance(term_results, dict):
        problems.append(process_text._generated_problem(
            "invalid_grouped_term_results",
            "Grouped term results are missing or malformed",
            "term_results must be an object keyed by the exact decimal rank "
            "strings from this source chunk.",
            path=f"$.{_GROUPED_TERM_RESULTS_KEY}",
            location="Response → term_results",
            expected={
                rank: "one grouped result object"
                for rank in sorted(expected_rank_keys, key=int)
            },
            actual=(
                "Field omitted"
                if term_results is missing_marker
                else type(term_results).__name__),
            suggestion=(
                "Retry using the frozen grouped structured-output schema.")))
        term_results = {}
    else:
        actual_rank_keys = set(term_results)
        if actual_rank_keys != expected_rank_keys:
            problems.append(process_text._generated_problem(
                "invalid_grouped_source_ranks",
                "Grouped term-result ranks do not match the request",
                "term_results must contain each requested rank exactly once "
                "and no other rank keys.",
                path=f"$.{_GROUPED_TERM_RESULTS_KEY}",
                location="Response → term_results",
                expected=sorted(expected_rank_keys, key=int),
                actual=sorted(actual_rank_keys),
                suggestion=(
                    "Retry so the result object uses exactly the requested "
                    "decimal rank keys.")))

    term_field = pipeline_store.get_language(
        pipeline.language_key).term_field
    expected_group_keys = {
        _GROUPED_CONTEXTUAL_SENSE_KEY,
        _GROUPED_ADDITIONAL_SENSES_KEY,
    }

    def inject_term(raw_sense, *, word, path, role):
        card_index = len(cards)
        if not isinstance(raw_sense, dict):
            problems.append(process_text._generated_problem(
                "invalid_grouped_source_sense",
                "Grouped source sense is not an object",
                f"Every {role} must be a JSON object containing the selected "
                "card fields, without repeating the term.",
                path=path,
                location=(
                    f'Response → term_results → rank {word.rank} → {role}'),
                scope="card",
                card_index=card_index,
                term=word.surface,
                expected="A JSON object.",
                actual=type(raw_sense).__name__,
                suggestion=(
                    "Retry using the frozen grouped structured-output "
                    "schema."),
                identity={
                    "rank": word.rank,
                    "role": role,
                }))
            return raw_sense
        card = dict(raw_sense)
        if term_field in card:
            problems.append(process_text._generated_problem(
                "grouped_source_sense_repeats_term",
                "Grouped source sense repeats its term",
                "V8 sense objects must not echo the term. The immutable rank "
                "mapping supplies the exact audited spelling.",
                path=(
                    f"{path}[{json.dumps(term_field, ensure_ascii=False)}]"),
                location=(
                    f'Response → term_results → rank {word.rank} → {role} '
                    f"→ {term_field}"),
                scope="field",
                card_index=card_index,
                term=word.surface,
                field_name=term_field,
                expected="Field omitted; rank supplies the term.",
                actual=card[term_field],
                suggestion=(
                    "Retry using the frozen grouped structured-output "
                    "schema."),
                identity={
                    "rank": word.rank,
                    "role": role,
                }))
        card[term_field] = word.surface
        return card

    # Iterate the immutable request order, not model-controlled object order.
    for rank_key, word in words_by_rank.items():
        if rank_key not in term_results:
            continue
        group = term_results[rank_key]
        group_path = (
            f"$.{_GROUPED_TERM_RESULTS_KEY}"
            f"[{json.dumps(rank_key)}]")
        if not isinstance(group, dict):
            problems.append(process_text._generated_problem(
                "invalid_grouped_term_result",
                "Grouped term result is not an object",
                "Each rank value must contain exactly contextual_sense and "
                "additional_senses.",
                path=group_path,
                location=(
                    f"Response → term_results → rank {word.rank}"),
                expected=sorted(expected_group_keys),
                actual=type(group).__name__,
                suggestion=(
                    "Retry using the frozen grouped structured-output "
                    "schema."),
                identity=word.rank))
            continue
        group_keys = set(group)
        if group_keys != expected_group_keys:
            problems.append(process_text._generated_problem(
                "invalid_grouped_term_result_fields",
                "Grouped term result has the wrong fields",
                "Each rank value must contain exactly contextual_sense and "
                "additional_senses.",
                path=group_path,
                location=(
                    f"Response → term_results → rank {word.rank}"),
                expected=sorted(expected_group_keys),
                actual=sorted(group_keys),
                suggestion=(
                    "Retry using the frozen grouped structured-output "
                    "schema."),
                identity=word.rank))

        contextual_path = (
            f"{group_path}.{_GROUPED_CONTEXTUAL_SENSE_KEY}")
        contextual = group.get(
            _GROUPED_CONTEXTUAL_SENSE_KEY,
            missing_marker)
        if contextual is missing_marker:
            problems.append(process_text._generated_problem(
                "missing_grouped_contextual_sense",
                "Grouped term has no contextual sense",
                "Every requested rank needs exactly one contextual_sense "
                "object.",
                path=contextual_path,
                location=(
                    f"Response → term_results → rank {word.rank} "
                    "→ contextual_sense"),
                expected="A JSON object.",
                actual="Field omitted",
                suggestion=(
                    "Retry so every requested rank has one contextual "
                    "sense."),
                identity=word.rank))
        else:
            contextual_index = len(cards)
            if isinstance(contextual, dict):
                for field_name, expected_value in (
                        process_text.grouped_contextual_field_constants(
                            pipeline,
                            word,
                            contexts_by_id).items()):
                    if field_name not in contextual:
                        continue
                    actual_value = contextual.get(
                        field_name,
                        missing_marker)
                    if actual_value == expected_value:
                        continue
                    field_path = (
                        f"{contextual_path}"
                        f"[{json.dumps(field_name, ensure_ascii=False)}]")
                    problems.append(process_text._generated_problem(
                        "contextual_source_field_mismatch",
                        "Contextual field contradicts the audited source",
                        "This exact retained occurrence has an unequivocal "
                        "source-specific lexical value frozen into the "
                        "current response contract.",
                        path=field_path,
                        location=(
                            f"Response → term_results → rank {word.rank} "
                            f"→ contextual_sense → {field_name}"),
                        scope="field",
                        card_index=contextual_index,
                        term=word.surface,
                        field_name=field_name,
                        expected=expected_value,
                        actual=(
                            "Field omitted"
                            if actual_value is missing_marker
                            else actual_value),
                        suggestion=(
                            "Retry using the frozen per-rank schema; do not "
                            "override the audited contextual grammar."),
                        identity={
                            "rank": word.rank,
                            "field": field_name,
                        }))
            cards.append(inject_term(
                contextual,
                word=word,
                path=contextual_path,
                role="contextual_sense"))
            card_raw_paths.append(contextual_path)
            contextual_indices.add(contextual_index)

        additional_path = (
            f"{group_path}.{_GROUPED_ADDITIONAL_SENSES_KEY}")
        additional = group.get(
            _GROUPED_ADDITIONAL_SENSES_KEY,
            missing_marker)
        if not isinstance(additional, list):
            problems.append(process_text._generated_problem(
                "invalid_grouped_additional_senses",
                "Grouped additional senses are missing or malformed",
                "additional_senses must be a JSON array; use an empty array "
                "when the term has no other common sense.",
                path=additional_path,
                location=(
                    f"Response → term_results → rank {word.rank} "
                    "→ additional_senses"),
                expected="A JSON array.",
                actual=(
                    "Field omitted"
                    if additional is missing_marker
                    else type(additional).__name__),
                suggestion=(
                    "Retry using the frozen grouped structured-output "
                    "schema."),
                identity=word.rank))
            continue
        for sense_index, raw_sense in enumerate(additional):
            card_index = len(cards)
            cards.append(inject_term(
                raw_sense,
                word=word,
                path=f"{additional_path}[{sense_index}]",
                role=f"additional_senses item {sense_index + 1}"))
            card_raw_paths.append(
                f"{additional_path}[{sense_index}]")
            additional_indices.add(card_index)

    raw_translations = parsed.get(translation_key, missing_marker)
    expected_context_ids = {
        context.context_id
        for context in chunk.contexts
    }
    source_language = pipeline_store.get_language(
        pipeline.language_key)
    requires_english_translation = (
        pipeline_store.translation_target_allowed(
            pipeline.language_key,
            "english"))
    translation_path = f"$.{translation_key}"
    if not isinstance(raw_translations, dict):
        problems.append(process_text._generated_problem(
            "invalid_grouped_source_context_translations",
            "Grouped source-context translations are missing or malformed",
            "source_context_translations must be an object whose exact keys "
            "are the context IDs retained in this source chunk.",
            path=translation_path,
            location="Response → source_context_translations",
            expected={
                context_id: "one complete English translation"
                for context_id in sorted(expected_context_ids)
            },
            actual=(
                "Field omitted"
                if raw_translations is missing_marker
                else type(raw_translations).__name__),
            suggestion=(
                "Retry using the frozen grouped structured-output schema.")))
        raw_translations = {}
    else:
        actual_context_ids = set(raw_translations)
        if actual_context_ids != expected_context_ids:
            problems.append(process_text._generated_problem(
                "invalid_grouped_source_context_ids",
                "Source-context translation IDs do not match the request",
                "source_context_translations must contain each retained "
                "context ID exactly once and no other keys.",
                path=translation_path,
                location="Response → source_context_translations",
                expected=sorted(expected_context_ids),
                actual=sorted(actual_context_ids),
                suggestion=(
                    "Retry so translations are keyed by exactly the retained "
                    "context IDs.")))

    # Validate each root value once.  A shared context may feed several cards,
    # but should never produce duplicate diagnostics.
    for context_id in sorted(expected_context_ids):
        if context_id not in raw_translations:
            continue
        raw_translation = raw_translations[context_id]
        item_path = (
            f"{translation_path}"
            f"[{json.dumps(context_id, ensure_ascii=False)}]")
        if include_source_context_nuance:
            expected_fields = {
                process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME,
                process_text.SOURCE_CONTEXT_NUANCE_FIELD_NAME,
            }
            if (
                    not isinstance(raw_translation, dict)
                    or set(raw_translation) != expected_fields):
                problems.append(process_text._generated_problem(
                    "invalid_source_context_translation_entry",
                    "Source-context entry is malformed",
                    "Every source-context value must contain exactly "
                    "translation and nuance.",
                    path=item_path,
                    location=(
                        "Response → source_context_translations "
                        f"→ {context_id}"),
                    expected=sorted(expected_fields),
                    actual=raw_translation,
                    suggestion=(
                        "Retry using the frozen grouped structured-output "
                        "schema."),
                    identity=context_id))
                continue
            translation = raw_translation[
                process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME]
            nuance = raw_translation[
                process_text.SOURCE_CONTEXT_NUANCE_FIELD_NAME]
            if not isinstance(nuance, str):
                problems.append(process_text._generated_problem(
                    "invalid_source_context_nuance",
                    "Source-context nuance is not text",
                    "The optional cultural or historical nuance must be a "
                    "JSON string; use an empty string when none is needed.",
                    path=(
                        f"{item_path}."
                        + process_text.SOURCE_CONTEXT_NUANCE_FIELD_NAME),
                    location=(
                        "Response → source_context_translations "
                        f"→ {context_id} → nuance"),
                    expected="A JSON string.",
                    actual=type(nuance).__name__,
                    suggestion=(
                        "Retry with a plain nuance string."),
                    identity=context_id))
            else:
                nuances[context_id] = nuance
                if _HTML_TAG_PATTERN.search(nuance):
                    problems.append(process_text._generated_problem(
                        "source_context_nuance_contains_html",
                        "Source-context nuance contains HTML",
                        "Sentence nuance must be plain text without HTML.",
                        path=(
                            f"{item_path}."
                            + process_text.SOURCE_CONTEXT_NUANCE_FIELD_NAME),
                        location=(
                            "Response → source_context_translations "
                            f"→ {context_id} → nuance"),
                        expected="Plain text without HTML.",
                        actual=nuance,
                        suggestion="Remove the HTML after checking the text.",
                        identity=context_id))
        else:
            translation = raw_translation
        if not isinstance(translation, str):
            problems.append(process_text._generated_problem(
                "invalid_source_context_translation_entry",
                "Source-context translation is not text",
                "Every source-context translation value must be a JSON "
                "string.",
                path=item_path,
                location=(
                    "Response → source_context_translations "
                    f"→ {context_id}"),
                expected="A JSON string.",
                actual=type(translation).__name__,
                suggestion=(
                    "Retry using the frozen grouped structured-output "
                    "schema."),
                identity=context_id,
                exception_type="TypeError"))
            continue
        translations[context_id] = translation
        if not translation.strip():
            problems.append(process_text._generated_problem(
                "blank_source_context_translation",
                "Source-context translation is blank",
                "Every retained source context needs one complete, non-blank "
                "English translation.",
                path=item_path,
                location=(
                    "Response → source_context_translations "
                    f"→ {context_id}"),
                expected="One complete English translation.",
                actual="Blank text",
                suggestion=(
                    "Retry so the retained context is translated before "
                    "packaging."),
                identity=context_id))
        if requires_english_translation:
            context = contexts_by_id.get(context_id)
            source_text = (
                context.text
                if context is not None
                else "")
            reason, plain_source = (
                process_text._sentence_translation_language_issue(
                    source_text,
                    translation,
                    source_language.model_language_key))
            if reason is not None:
                if reason == "exact_source_copy":
                    message = (
                        "This English context translation is an exact copy "
                        "of the retained source-language passage.")
                else:
                    message = (
                        "This context translation is required to be English, "
                        f"but it contains {source_language.name} script and "
                        "no Latin alphabetic text.")
                problems.append(process_text._generated_problem(
                    "source_context_translation_not_english",
                    "Source-context translation is not English",
                    message,
                    path=item_path,
                    location=(
                        "Response → source_context_translations "
                        f"→ {context_id}"),
                    expected=(
                        "One complete natural English translation of the "
                        "retained source context."),
                    actual={
                        "translation": translation,
                        "source_context": plain_source,
                        "reason": reason,
                    },
                    suggestion=(
                        "Retry or replace this value with an English "
                        "translation of the complete retained context."),
                    identity={
                        "context_id": context_id,
                        "reason": reason,
                    }))
        if "|" in html.unescape(translation):
            problems.append(process_text._generated_problem(
                "sentence_translation_count_mismatch",
                "Source-context translation contains the card delimiter",
                "A root source-context value is one complete translation. It "
                "cannot contain | because that character separates generated "
                "example translations on an Anki card.",
                path=item_path,
                location=(
                    "Response → source_context_translations "
                    f"→ {context_id}"),
                expected="One complete translation without |.",
                actual=translation,
                suggestion=(
                    "Retry or replace the delimiter with normal "
                    "punctuation."),
                identity=context_id))
        if _HTML_TAG_PATTERN.search(translation):
            problems.append(process_text._generated_problem(
                "source_context_translation_contains_html",
                "Source-context translation contains HTML",
                "A retained context translation must be plain English text "
                "without HTML.",
                path=item_path,
                location=(
                    "Response → source_context_translations "
                    f"→ {context_id}"),
                expected="Plain text without HTML.",
                actual=translation,
                suggestion=(
                    "Retry or remove the HTML tags after checking the "
                    "translation."),
                identity=context_id))

    return (
        cards,
        card_raw_paths,
        translations,
        nuances,
        contextual_indices,
        additional_indices,
        problems,
    )


def _mark_compact_problem(
        problem,
        *,
        source_rank=None,
        context_id=None,
        full_retry_required=False):
    """Attach the smallest safe compact-protocol repair scope."""
    if source_rank is not None:
        problem["source_rank"] = source_rank
    if context_id is not None:
        problem["context_id"] = context_id
    if full_retry_required or problem.get("full_retry_required"):
        problem["full_retry_required"] = True
        problem.pop("repair_target", None)
    elif source_rank is not None:
        problem["repair_target"] = {
            "kind": "source_rank",
            "source_rank": source_rank,
        }
    elif context_id is not None:
        problem["repair_target"] = {
            "kind": "context",
            "context_id": context_id,
        }
    return problem


def _compact_problem(
        code,
        title,
        message,
        *,
        source_rank=None,
        context_id=None,
        full_retry_required=False,
        **kwargs):
    return _mark_compact_problem(
        process_text._generated_problem(
            code,
            title,
            message,
            **kwargs),
        source_rank=source_rank,
        context_id=context_id,
        full_retry_required=full_retry_required)


def _normalize_compact_source_results(
        parsed,
        pipeline,
        chunk,
        *,
        enforce_contextual_constants=True,
        source_context_translation_memory=None,
        include_source_context_nuance=False):
    """Normalize compact result arrays into canonical card candidates."""
    problems = []
    cards = []
    card_raw_paths = []
    card_source_ranks = []
    card_context_ids = []
    contextual_indices = set()
    additional_indices = set()
    translations = {}
    nuances = {}
    missing_marker = object()

    term_results_key = process_text.SOURCE_TERM_RESULTS_KEY
    translation_key = process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY
    contextual_key = process_text.SOURCE_CONTEXTUAL_SENSE_KEY
    additional_key = process_text.SOURCE_ADDITIONAL_SENSES_KEY
    context_id_key = process_text.SOURCE_CONTEXT_ID_FIELD_NAME
    context_translation_key = (
        process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME)
    expected_root_keys = {
        term_results_key,
        translation_key,
    }

    if not isinstance(parsed, dict):
        problems.append(_compact_problem(
            "invalid_compact_source_response_root",
            "Compact source response root is malformed",
            "A compact source response must be one JSON object containing "
            "only term_results and source_context_translations.",
            path="$",
            location="Response",
            expected=sorted(expected_root_keys),
            actual=type(parsed).__name__,
            suggestion=(
                "Retry using the frozen compact structured-output schema."),
            full_retry_required=True))
        return (
            cards,
            card_raw_paths,
            card_source_ranks,
            card_context_ids,
            translations,
            nuances,
            contextual_indices,
            additional_indices,
            problems,
        )

    root_keys = set(parsed)
    if root_keys != expected_root_keys:
        problems.append(_compact_problem(
            "invalid_compact_source_response_root",
            "Compact source response root has the wrong fields",
            "A compact source response must contain exactly term_results and "
            "source_context_translations.",
            path="$",
            location="Response",
            expected=sorted(expected_root_keys),
            actual=sorted(root_keys),
            suggestion=(
                "Retry using the frozen compact structured-output schema."),
            full_retry_required=True))

    words_by_rank = {}
    for word in chunk.words:
        rank = word.rank
        if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank <= 0):
            problems.append(_compact_problem(
                "invalid_requested_source_rank",
                "The source chunk contains an invalid rank",
                "Every requested source rank must be a positive integer.",
                path=f"request.words[rank={rank}]",
                location=f'Requested term “{word.surface}”',
                scope="request",
                term=word.surface,
                expected="A positive non-boolean integer.",
                actual=rank,
                suggestion=(
                    "Create a new source job from a valid audited plan."),
                identity=rank,
                full_retry_required=True))
            continue
        if rank in words_by_rank:
            problems.append(_compact_problem(
                "duplicate_requested_source_rank",
                "The source chunk repeats a requested rank",
                "Every compact result must be addressable by one unique "
                "source rank.",
                path=f"request.words[rank={rank}]",
                location=f'Requested term “{word.surface}”',
                scope="request",
                term=word.surface,
                expected="A rank unique within this source chunk.",
                actual=rank,
                suggestion=(
                    "Create a new source job from a valid audited plan."),
                identity=rank,
                source_rank=rank,
                full_retry_required=True))
            continue
        words_by_rank[rank] = word
    expected_ranks = tuple(sorted(words_by_rank))
    expected_rank_set = set(expected_ranks)

    contexts_by_id = {}
    for context in chunk.contexts:
        context_id = context.context_id
        if context_id in contexts_by_id:
            problems.append(_compact_problem(
                "duplicate_requested_source_context",
                "The source chunk repeats a context ID",
                "Every retained context must have one unique context_id.",
                path=f"request.contexts[{json.dumps(context_id)}]",
                location=f"Requested context {context_id}",
                scope="request",
                expected="A context_id unique within this source chunk.",
                actual=context_id,
                suggestion=(
                    "Create a new source job from a valid audited plan."),
                identity=context_id,
                context_id=context_id,
                full_retry_required=True))
            continue
        contexts_by_id[context_id] = context
    expected_context_ids = set(contexts_by_id)
    if source_context_translation_memory is None:
        source_context_translation_memory = {}
    if not isinstance(source_context_translation_memory, dict):
        raise TypeError(
            "Source-context translation memory must be an object.")
    for context_id, hit in source_context_translation_memory.items():
        translation = (
            hit.get("translation")
            if isinstance(hit, dict)
            else hit)
        if (
                context_id not in expected_context_ids
                or not isinstance(translation, str)
                or not translation.strip()
                or "|" in html.unescape(translation)
                or _HTML_TAG_PATTERN.search(translation)):
            raise ValueError(
                "Source-context translation memory does not match this "
                "chunk.")
        translations[context_id] = translation
    remembered_context_ids = set(translations)
    expected_provider_context_ids = (
        expected_context_ids - remembered_context_ids)

    term_results = parsed.get(term_results_key, missing_marker)
    term_results_path = f"$.{term_results_key}"
    indexed_results = {}
    rank_sequence = []
    seen_ranks = set()
    expected_item_keys = {
        _COMPACT_RANK_FIELD,
        contextual_key,
        additional_key,
    }
    if not isinstance(term_results, list):
        problems.append(_compact_problem(
            "invalid_compact_term_results",
            "Compact term results are missing or malformed",
            "term_results must be an array containing one item per requested "
            "positive integer rank.",
            path=term_results_path,
            location="Response → term_results",
            expected="An ascending array containing each requested rank once.",
            actual=(
                "Field omitted"
                if term_results is missing_marker
                else type(term_results).__name__),
            suggestion=(
                "Retry using the frozen compact structured-output schema."),
            full_retry_required=True))
        term_results = []

    for result_index, result in enumerate(term_results):
        result_path = f"{term_results_path}[{result_index}]"
        if not isinstance(result, dict):
            problems.append(_compact_problem(
                "invalid_compact_term_result",
                "Compact term result is not an object",
                "Every term_results item must contain exactly rank, "
                "contextual_sense, and additional_senses.",
                path=result_path,
                location=f"Response → term_results → item {result_index + 1}",
                expected=sorted(expected_item_keys),
                actual=type(result).__name__,
                suggestion=(
                    "Retry using the frozen compact structured-output "
                    "schema."),
                identity=result_index,
                full_retry_required=True))
            continue

        rank = result.get(_COMPACT_RANK_FIELD, missing_marker)
        valid_rank = (
            not isinstance(rank, bool)
            and isinstance(rank, int)
            and rank > 0)
        known_rank = valid_rank and rank in expected_rank_set
        word = words_by_rank.get(rank) if known_rank else None
        context_id = word.context_id if word is not None else None
        if set(result) != expected_item_keys:
            problems.append(_compact_problem(
                "invalid_compact_term_result_fields",
                "Compact term result has the wrong fields",
                "Every term_results item must contain exactly rank, "
                "contextual_sense, and additional_senses.",
                path=result_path,
                location=f"Response → term_results → item {result_index + 1}",
                expected=sorted(expected_item_keys),
                actual=sorted(result),
                suggestion=(
                    "Retry using the frozen compact structured-output "
                    "schema."),
                identity={
                    "result_index": result_index,
                    "rank": (
                        rank
                        if rank is not missing_marker
                        else None),
                },
                source_rank=rank if known_rank else None,
                context_id=context_id,
                full_retry_required=not known_rank))
        if not valid_rank:
            problems.append(_compact_problem(
                "invalid_compact_source_rank",
                "Compact source rank is invalid",
                "Every term_results rank must be a positive non-boolean "
                "integer.",
                path=(
                    f"{result_path}.{_COMPACT_RANK_FIELD}"),
                location=(
                    "Response → term_results "
                    f"→ item {result_index + 1} → rank"),
                expected="A positive non-boolean integer.",
                actual=(
                    "Field omitted"
                    if rank is missing_marker
                    else rank),
                suggestion=(
                    "Retry using the frozen compact structured-output "
                    "schema."),
                identity=result_index,
                full_retry_required=True))
            continue

        rank_sequence.append(rank)
        if rank not in expected_rank_set:
            problems.append(_compact_problem(
                "unexpected_compact_source_rank",
                "Compact response contains an unrequested rank",
                "Every term_results rank must identify a term in this source "
                "chunk.",
                path=f"{result_path}.{_COMPACT_RANK_FIELD}",
                location=(
                    "Response → term_results "
                    f"→ item {result_index + 1} → rank"),
                expected=list(expected_ranks),
                actual=rank,
                suggestion=(
                    "Retry without the unrequested term result."),
                identity=rank,
                source_rank=rank,
                full_retry_required=True))
        if rank in seen_ranks:
            problems.append(_compact_problem(
                "duplicate_compact_source_rank",
                "Compact response repeats a source rank",
                "Each requested rank must occur exactly once in term_results.",
                path=f"{result_path}.{_COMPACT_RANK_FIELD}",
                location=(
                    "Response → term_results "
                    f"→ item {result_index + 1} → rank"),
                expected="A rank not used by an earlier item.",
                actual=rank,
                suggestion=(
                    "Retry or replace the duplicate term result."),
                identity={
                    "rank": rank,
                    "result_index": result_index,
                },
                source_rank=rank if known_rank else None,
                context_id=context_id,
                full_retry_required=not known_rank))
            continue
        seen_ranks.add(rank)
        if known_rank:
            indexed_results[rank] = (result_index, result)

    if any(
            previous > current
            for previous, current in zip(
                rank_sequence,
                rank_sequence[1:])):
        problems.append(_compact_problem(
            "compact_source_ranks_not_ascending",
            "Compact source ranks are not strictly ascending",
            "term_results must be ordered by increasing rank so array "
            "positions remain deterministic.",
            path=term_results_path,
            location="Response → term_results",
            expected=sorted(rank_sequence),
            actual=rank_sequence,
            suggestion=(
                "Retry with term_results sorted by rank."),
            identity=rank_sequence,
            full_retry_required=True))

    for rank in expected_ranks:
        if rank in indexed_results:
            continue
        word = words_by_rank[rank]
        problems.append(_compact_problem(
            "missing_compact_source_rank",
            "Compact response omitted a requested rank",
            "Each requested rank must occur exactly once in term_results.",
            path=term_results_path,
            location=f'Requested term “{word.surface}”',
            scope="request",
            term=word.surface,
            expected=rank,
            actual="No matching term_results item.",
            suggestion=(
                "Repair this rank or retry the source request."),
            identity=rank,
            source_rank=rank,
            context_id=word.context_id))

    term_field = pipeline_store.get_language(
        pipeline.language_key).term_field

    def inject_term(raw_sense, *, word, path, role):
        card_index = len(cards)
        if not isinstance(raw_sense, dict):
            problems.append(_compact_problem(
                "invalid_compact_source_sense",
                "Compact source sense is not an object",
                f"Every {role} must be a JSON object containing the selected "
                "card fields, without repeating the term.",
                path=path,
                location=(
                    f"Response → term_results → rank {word.rank} → {role}"),
                scope="card",
                card_index=card_index,
                term=word.surface,
                expected="A JSON object.",
                actual=type(raw_sense).__name__,
                suggestion=(
                    "Repair this rank using the frozen compact schema."),
                identity={
                    "rank": word.rank,
                    "role": role,
                },
                source_rank=word.rank,
                context_id=word.context_id))
            return raw_sense
        card = dict(raw_sense)
        if term_field in card:
            problems.append(_compact_problem(
                "compact_source_sense_repeats_term",
                "Compact source sense repeats its term",
                "V9 sense objects must not echo the term. The immutable rank "
                "mapping supplies the exact audited spelling.",
                path=(
                    f"{path}[{json.dumps(term_field, ensure_ascii=False)}]"),
                location=(
                    f"Response → term_results → rank {word.rank} → {role} "
                    f"→ {term_field}"),
                scope="field",
                card_index=card_index,
                term=word.surface,
                field_name=term_field,
                expected="Field omitted; rank supplies the term.",
                actual=card[term_field],
                suggestion=(
                    "Repair this rank using the frozen compact schema."),
                identity={
                    "rank": word.rank,
                    "role": role,
                },
                source_rank=word.rank,
                context_id=word.context_id))
        card[term_field] = word.surface
        return card

    # Array order is checked above; immutable ranks still control term identity
    # and canonical ordering.
    for rank in expected_ranks:
        indexed_result = indexed_results.get(rank)
        if indexed_result is None:
            continue
        result_index, result = indexed_result
        word = words_by_rank[rank]
        result_path = f"{term_results_path}[{result_index}]"

        contextual_path = f"{result_path}.{contextual_key}"
        contextual = result.get(contextual_key, missing_marker)
        if contextual is missing_marker:
            problems.append(_compact_problem(
                "missing_compact_contextual_sense",
                "Compact term has no contextual sense",
                "Every requested rank needs exactly one contextual_sense "
                "object.",
                path=contextual_path,
                location=(
                    f"Response → term_results → rank {rank} "
                    "→ contextual_sense"),
                expected="A JSON object.",
                actual="Field omitted",
                suggestion=(
                    "Repair this rank with one contextual sense."),
                identity=rank,
                source_rank=rank,
                context_id=word.context_id))
        else:
            contextual_index = len(cards)
            if (
                    isinstance(contextual, dict)
                    and enforce_contextual_constants):
                for field_name, expected_value in (
                        process_text.compact_contextual_field_constants(
                            pipeline,
                            word,
                            contexts_by_id).items()):
                    if field_name not in contextual:
                        continue
                    actual_value = contextual.get(
                        field_name,
                        missing_marker)
                    if actual_value == expected_value:
                        continue
                    problems.append(_compact_problem(
                        "contextual_source_field_mismatch",
                        "Contextual field contradicts the audited source",
                        "This exact retained occurrence has an unequivocal "
                        "source-specific lexical value frozen into the "
                        "current response contract.",
                        path=(
                            f"{contextual_path}"
                            f"[{json.dumps(field_name, ensure_ascii=False)}]"),
                        location=(
                            f"Response → term_results → rank {rank} "
                            f"→ contextual_sense → {field_name}"),
                        scope="field",
                        card_index=contextual_index,
                        term=word.surface,
                        field_name=field_name,
                        expected=expected_value,
                        actual=(
                            "Field omitted"
                            if actual_value is missing_marker
                            else actual_value),
                        suggestion=(
                            "Repair this rank without overriding the audited "
                            "contextual grammar."),
                        identity={
                            "rank": rank,
                            "field": field_name,
                        },
                        source_rank=rank,
                        context_id=word.context_id))
            cards.append(inject_term(
                contextual,
                word=word,
                path=contextual_path,
                role="contextual_sense"))
            card_raw_paths.append(contextual_path)
            card_source_ranks.append(rank)
            card_context_ids.append(word.context_id)
            contextual_indices.add(contextual_index)

        additional_path = f"{result_path}.{additional_key}"
        additional = result.get(additional_key, missing_marker)
        if not isinstance(additional, list):
            problems.append(_compact_problem(
                "invalid_compact_additional_senses",
                "Compact additional senses are missing or malformed",
                "additional_senses must be a JSON array; use an empty array "
                "when the term has no other common sense.",
                path=additional_path,
                location=(
                    f"Response → term_results → rank {rank} "
                    "→ additional_senses"),
                expected="A JSON array.",
                actual=(
                    "Field omitted"
                    if additional is missing_marker
                    else type(additional).__name__),
                suggestion=(
                    "Repair this rank using the frozen compact schema."),
                identity=rank,
                source_rank=rank,
                context_id=word.context_id))
            continue
        for sense_index, raw_sense in enumerate(additional):
            card_index = len(cards)
            raw_path = f"{additional_path}[{sense_index}]"
            cards.append(inject_term(
                raw_sense,
                word=word,
                path=raw_path,
                role=f"additional_senses item {sense_index + 1}"))
            card_raw_paths.append(raw_path)
            card_source_ranks.append(rank)
            card_context_ids.append(word.context_id)
            additional_indices.add(card_index)

    raw_translations = parsed.get(translation_key, missing_marker)
    translation_path = f"$.{translation_key}"
    expected_translation_keys = {
        context_id_key,
        context_translation_key,
    }
    if include_source_context_nuance:
        expected_translation_keys.add(
            process_text.SOURCE_CONTEXT_NUANCE_FIELD_NAME)
    if not isinstance(raw_translations, list):
        problems.append(_compact_problem(
            "invalid_compact_source_context_translations",
            "Compact source-context translations are missing or malformed",
            "source_context_translations must be an array containing one "
            "exact two-field item per retained context.",
            path=translation_path,
            location="Response → source_context_translations",
            expected=(
                "An array containing each non-remembered context_id exactly "
                "once."),
            actual=(
                "Field omitted"
                if raw_translations is missing_marker
                else type(raw_translations).__name__),
            suggestion=(
                "Retry using the frozen compact structured-output schema."),
            full_retry_required=True))
        raw_translations = []

    source_language = pipeline_store.get_language(
        pipeline.language_key)
    requires_english_translation = (
        pipeline_store.translation_target_allowed(
            pipeline.language_key,
            "english"))
    seen_context_ids = set()
    for entry_index, entry in enumerate(raw_translations):
        entry_path = f"{translation_path}[{entry_index}]"
        if not isinstance(entry, dict):
            problems.append(_compact_problem(
                "invalid_compact_source_context_translation_entry",
                "Source-context translation entry is malformed",
                "Every source-context translation must contain exactly the "
                "text fields context_id and translation.",
                path=entry_path,
                location=(
                    "Response → source_context_translations "
                    f"→ item {entry_index + 1}"),
                expected={
                    context_id_key: "string",
                    context_translation_key: "string",
                },
                actual=entry,
                suggestion=(
                    "Retry using the frozen compact structured-output "
                    "schema."),
                identity=entry_index,
                full_retry_required=True))
            continue

        context_id = entry.get(context_id_key, missing_marker)
        known_context = (
            isinstance(context_id, str)
            and context_id in expected_context_ids)
        if set(entry) != expected_translation_keys:
            problems.append(_compact_problem(
                "invalid_compact_source_context_translation_fields",
                "Source-context translation entry has the wrong fields",
                "Every source-context translation must contain exactly the "
                "text fields context_id and translation.",
                path=entry_path,
                location=(
                    "Response → source_context_translations "
                    f"→ item {entry_index + 1}"),
                expected=sorted(expected_translation_keys),
                actual=sorted(entry),
                suggestion=(
                    "Repair this context or retry using the frozen compact "
                    "schema."),
                identity={
                    "entry_index": entry_index,
                    "context_id": (
                        context_id
                        if context_id is not missing_marker
                        else None),
                },
                context_id=context_id if known_context else None,
                full_retry_required=not known_context))
        if not isinstance(context_id, str):
            problems.append(_compact_problem(
                "invalid_compact_source_context_id",
                "Source-context translation ID is invalid",
                "Every context_id must be a JSON string matching a retained "
                "source context.",
                path=f"{entry_path}.{context_id_key}",
                location=(
                    "Response → source_context_translations "
                    f"→ item {entry_index + 1} → context_id"),
                expected=sorted(expected_context_ids),
                actual=(
                    "Field omitted"
                    if context_id is missing_marker
                    else context_id),
                suggestion=(
                    "Retry using the frozen compact structured-output "
                    "schema."),
                identity=entry_index,
                full_retry_required=True))
            continue
        if context_id not in expected_context_ids:
            problems.append(_compact_problem(
                "unexpected_compact_source_context_id",
                "An unrequested source context was translated",
                "Every context_id must identify a retained context in this "
                "source chunk.",
                path=f"{entry_path}.{context_id_key}",
                location=(
                    "Response → source_context_translations "
                    f"→ item {entry_index + 1} → context_id"),
                expected=sorted(expected_context_ids),
                actual=context_id,
                suggestion=(
                    "Retry without the unrequested context translation."),
                identity=context_id,
                context_id=context_id,
                full_retry_required=True))
        if (
                known_context
                and context_id in remembered_context_ids
                and set(entry) == expected_translation_keys
                and entry.get(context_translation_key)
                == translations[context_id]):
            # Echoing an exact trusted value wastes output tokens but does not
            # make an otherwise valid paid response unusable.
            continue
        if known_context and context_id in remembered_context_ids:
            problems.append(_compact_problem(
                "provider_overrode_translation_memory",
                "Provider contradicted a remembered source translation",
                "A context with an exact locally validated translation must "
                "be omitted from source_context_translations, not replaced.",
                path=entry_path,
                location=(
                    "Response → source_context_translations "
                    f"→ {context_id}"),
                expected="The remembered context_id omitted.",
                actual=entry,
                suggestion=(
                    "Remove this entry and retain the frozen local "
                    "translation."),
                identity=context_id,
                context_id=context_id))
            continue
        if context_id in seen_context_ids:
            problems.append(_compact_problem(
                "duplicate_compact_source_context_translation",
                "Source context was translated more than once",
                "Each requested context_id must occur exactly once in "
                "source_context_translations.",
                path=f"{entry_path}.{context_id_key}",
                location=(
                    "Response → source_context_translations "
                    f"→ item {entry_index + 1} → context_id"),
                expected="A context_id not used by an earlier item.",
                actual=context_id,
                suggestion=(
                    "Repair this context or remove the duplicate item."),
                identity={
                    "context_id": context_id,
                    "entry_index": entry_index,
                },
                context_id=context_id if known_context else None,
                full_retry_required=not known_context))
            continue
        seen_context_ids.add(context_id)
        if not known_context:
            continue

        translation = entry.get(
            context_translation_key,
            missing_marker)
        value_path = f"{entry_path}.{context_translation_key}"
        if not isinstance(translation, str):
            problems.append(_compact_problem(
                "invalid_source_context_translation_entry",
                "Source-context translation is not text",
                "Every source-context translation value must be a JSON "
                "string.",
                path=value_path,
                location=(
                    "Response → source_context_translations "
                    f"→ {context_id} → translation"),
                expected="A JSON string.",
                actual=(
                    "Field omitted"
                    if translation is missing_marker
                    else type(translation).__name__),
                suggestion=(
                    "Repair this context using the frozen compact schema."),
                identity=context_id,
                exception_type="TypeError",
                context_id=context_id))
            continue

        translations[context_id] = translation
        if include_source_context_nuance:
            nuance = entry.get(
                process_text.SOURCE_CONTEXT_NUANCE_FIELD_NAME,
                missing_marker)
            nuance_path = (
                f"{entry_path}."
                + process_text.SOURCE_CONTEXT_NUANCE_FIELD_NAME)
            if not isinstance(nuance, str):
                problems.append(_compact_problem(
                    "invalid_source_context_nuance",
                    "Source-context nuance is not text",
                    "The optional cultural or historical nuance must be a "
                    "JSON string; use an empty string when none is needed.",
                    path=nuance_path,
                    location=(
                        "Response → source_context_translations "
                        f"→ {context_id} → nuance"),
                    expected="A JSON string.",
                    actual=(
                        "Field omitted"
                        if nuance is missing_marker
                        else type(nuance).__name__),
                    suggestion="Repair this context with a plain text value.",
                    identity=context_id,
                    context_id=context_id))
            else:
                nuances[context_id] = nuance
                if _HTML_TAG_PATTERN.search(nuance):
                    problems.append(_compact_problem(
                        "source_context_nuance_contains_html",
                        "Source-context nuance contains HTML",
                        "Sentence nuance must be plain text without HTML.",
                        path=nuance_path,
                        location=(
                            "Response → source_context_translations "
                            f"→ {context_id} → nuance"),
                        expected="Plain text without HTML.",
                        actual=nuance,
                        suggestion=(
                            "Repair this context by removing the HTML."),
                        identity=context_id,
                        context_id=context_id))
        if not translation.strip():
            problems.append(_compact_problem(
                "blank_source_context_translation",
                "Source-context translation is blank",
                "Every retained source context needs one complete, non-blank "
                "English translation.",
                path=value_path,
                location=(
                    "Response → source_context_translations "
                    f"→ {context_id} → translation"),
                expected="One complete English translation.",
                actual="Blank text",
                suggestion=(
                    "Repair this context with a complete translation."),
                identity=context_id,
                context_id=context_id))
        if requires_english_translation:
            context = contexts_by_id[context_id]
            reason, plain_source = (
                process_text._sentence_translation_language_issue(
                    context.text,
                    translation,
                    source_language.model_language_key))
            if reason is not None:
                if reason == "exact_source_copy":
                    message = (
                        "This English context translation is an exact copy "
                        "of the retained source-language passage.")
                else:
                    message = (
                        "This context translation is required to be English, "
                        f"but it contains {source_language.name} script and "
                        "no Latin alphabetic text.")
                problems.append(_compact_problem(
                    "source_context_translation_not_english",
                    "Source-context translation is not English",
                    message,
                    path=value_path,
                    location=(
                        "Response → source_context_translations "
                        f"→ {context_id} → translation"),
                    expected=(
                        "One complete natural English translation of the "
                        "retained source context."),
                    actual={
                        "translation": translation,
                        "source_context": plain_source,
                        "reason": reason,
                    },
                    suggestion=(
                        "Repair this context with an English translation of "
                        "the complete retained context."),
                    identity={
                        "context_id": context_id,
                        "reason": reason,
                    },
                    context_id=context_id))
        if "|" in html.unescape(translation):
            problems.append(_compact_problem(
                "sentence_translation_count_mismatch",
                "Source-context translation contains the card delimiter",
                "A root source-context value is one complete translation. It "
                "cannot contain | because that character separates generated "
                "example translations on an Anki card.",
                path=value_path,
                location=(
                    "Response → source_context_translations "
                    f"→ {context_id} → translation"),
                expected="One complete translation without |.",
                actual=translation,
                suggestion=(
                    "Repair this context by replacing the delimiter with "
                    "normal punctuation."),
                identity=context_id,
                context_id=context_id))
        if _HTML_TAG_PATTERN.search(translation):
            problems.append(_compact_problem(
                "source_context_translation_contains_html",
                "Source-context translation contains HTML",
                "A retained context translation must be plain English text "
                "without HTML.",
                path=value_path,
                location=(
                    "Response → source_context_translations "
                    f"→ {context_id} → translation"),
                expected="Plain text without HTML.",
                actual=translation,
                suggestion=(
                    "Repair this context by removing the HTML tags after "
                    "checking the translation."),
                identity=context_id,
                context_id=context_id))

    for context_id in sorted(
            expected_provider_context_ids - set(translations)):
        problems.append(_compact_problem(
            "missing_source_context_translation",
            "A retained source context has no translation",
            "Every context_id supplied in this source chunk must have exactly "
            "one complete English translation.",
            path=translation_path,
            location=f"Requested context {context_id}",
            scope="request",
            expected=context_id,
            actual="No matching valid translation object.",
            suggestion=(
                "Repair this context with one complete English translation."),
            identity=context_id,
            context_id=context_id))

    return (
        cards,
        card_raw_paths,
        card_source_ranks,
        card_context_ids,
        translations,
        nuances,
        contextual_indices,
        additional_indices,
        problems,
    )


_CANONICAL_CARD_PATH_PATTERN = re.compile(
    r"^\$\.cards\[(?P<card_index>\d+)\](?P<suffix>.*)$")


def _remap_grouped_card_problem_paths(report, card_raw_paths):
    """Replace temporary ``$.cards`` paths with their raw v8 locations.

    Grouped responses are flattened only for reuse of the ordinary card
    validator.  The review report must continue to address the JSON the user
    and repair tools can actually inspect.  Problem IDs remain unchanged so
    persisted review decisions are not invalidated by this presentation fix.
    """
    for problem in report.get("problems", ()):
        path = problem.get("path")
        if not isinstance(path, str):
            continue
        match = _CANONICAL_CARD_PATH_PATTERN.fullmatch(path)
        if match is None:
            continue
        card_index = int(match.group("card_index"))
        if not 0 <= card_index < len(card_raw_paths):
            continue
        problem["path"] = (
            card_raw_paths[card_index]
            + match.group("suffix"))
    return report


_REQUEST_RANK_PATH_PATTERN = re.compile(
    r"^request\.words\[rank=(?P<rank>\d+)\]")


def _finalize_compact_report(
        report,
        card_raw_paths,
        card_source_ranks,
        card_context_ids):
    """Expose compact raw paths and propagate per-item repair metadata."""
    for problem in report.get("problems", ()):
        card_index = problem.get("card_index")
        if (
                isinstance(card_index, int)
                and not isinstance(card_index, bool)
                and 0 <= card_index < len(card_raw_paths)):
            path = problem.get("path")
            if isinstance(path, str):
                match = _CANONICAL_CARD_PATH_PATTERN.fullmatch(path)
                if match is not None:
                    problem["path"] = (
                        card_raw_paths[card_index]
                        + match.group("suffix"))
            _mark_compact_problem(
                problem,
                source_rank=card_source_ranks[card_index],
                context_id=card_context_ids[card_index])
            continue

        if (
                "repair_target" in problem
                or problem.get("full_retry_required")):
            continue
        path = problem.get("path")
        rank_match = (
            _REQUEST_RANK_PATH_PATTERN.match(path)
            if isinstance(path, str)
            else None)
        if (
                rank_match is not None
                and problem.get("code") == "missing_source_term"):
            _mark_compact_problem(
                problem,
                source_rank=int(rank_match.group("rank")))
            continue
        # A compact response problem that cannot be traced to one known result or
        # retained context must not be sent to a selective repair request.
        if problem.get("scope") != "request":
            _mark_compact_problem(
                problem,
                full_retry_required=True)

    report["full_retry_required"] = any(
        problem.get("full_retry_required", False)
        for problem in report.get("problems", ()))
    return report


def _split_source_card_collections(parsed, pipeline, chunk):
    """Normalize the v6 split roots into contextual-first internal cards."""
    problems = []
    contextual_key = process_text.SOURCE_CONTEXTUAL_CARDS_KEY
    additional_key = process_text.SOURCE_ADDITIONAL_SENSE_CARDS_KEY
    translation_key = process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY
    expected_root_keys = {
        contextual_key,
        additional_key,
        translation_key,
    }
    missing_marker = object()

    if not isinstance(parsed, dict):
        problems.append(process_text._generated_problem(
            "invalid_split_source_response_root",
            "Split source response root is malformed",
            "Current source responses must be one JSON object containing "
            "the contextual, additional-sense, and context-translation "
            "arrays.",
            path="$",
            location="Response",
            expected=sorted(expected_root_keys),
            actual=type(parsed).__name__,
            suggestion=(
                "Retry using the frozen structured-output schema.")))
        return [], [], problems

    root_keys = set(parsed)
    if root_keys != expected_root_keys:
        problems.append(process_text._generated_problem(
            "invalid_split_source_response_root",
            "Split source response root has the wrong fields",
            "Current source responses must contain exactly the three split "
            "root arrays required by the frozen schema.",
            path="$",
            location="Response",
            expected=sorted(expected_root_keys),
            actual=sorted(root_keys),
            suggestion=(
                "Retry using the frozen structured-output schema.")))

    raw_contextual = parsed.pop(contextual_key, missing_marker)
    raw_additional = parsed.pop(additional_key, missing_marker)

    def collection(value, key):
        if isinstance(value, list):
            return value
        problems.append(process_text._generated_problem(
            "invalid_split_source_card_collection",
            "Split source card collection is missing or malformed",
            f"{key} must be a JSON array.",
            path=f"$.{key}",
            location=f"Response → {key}",
            expected="A JSON array.",
            actual=(
                "Field omitted"
                if value is missing_marker
                else type(value).__name__),
            suggestion=(
                "Retry using the frozen structured-output schema."),
            identity=key))
        return []

    contextual_cards = collection(raw_contextual, contextual_key)
    additional_cards = collection(raw_additional, additional_key)
    term_field = pipeline_store.get_language(
        pipeline.language_key).term_field
    requested_order = []
    requested_terms = {}
    for word in chunk.words:
        normalized = _term_identity(word.surface, pipeline)
        if normalized not in requested_terms:
            requested_order.append(normalized)
            requested_terms[normalized] = word

    contextual_counts = Counter()
    for card_index, card in enumerate(contextual_cards):
        if (
                not isinstance(card, dict)
                or not isinstance(card.get(term_field), str)):
            # Ordinary schema validation reports the malformed card/field.
            continue
        term = card[term_field]
        normalized = _term_identity(term, pipeline)
        contextual_counts[normalized] += 1
        if normalized not in requested_terms:
            problems.append(process_text._generated_problem(
                "unexpected_contextual_source_term",
                "Contextual card term was not requested",
                "contextual_cards may contain exactly one card for each "
                "requested normalized term and no other terms.",
                path=(
                    f"$.{contextual_key}[{card_index}].{term_field}"),
                location=(
                    f"Response → {contextual_key} → item "
                    f"{card_index + 1} → {term_field}"),
                scope="field",
                card_index=card_index,
                term=term,
                field_name=term_field,
                expected=[
                    requested_terms[item].surface
                    for item in requested_order
                ],
                actual=term,
                suggestion=(
                    "Retry and remove every unrequested contextual card."),
                identity=normalized))

    for normalized in requested_order:
        word = requested_terms[normalized]
        count = contextual_counts[normalized]
        if count == 1:
            continue
        code = (
            "missing_contextual_source_card"
            if count == 0
            else "duplicate_contextual_source_card")
        title = (
            "Requested term has no contextual card"
            if count == 0
            else "Requested term has multiple contextual cards")
        problems.append(process_text._generated_problem(
            code,
            title,
            "contextual_cards must contain exactly one card for every "
            "requested normalized term.",
            path=f"$.{contextual_key}",
            location=f'Requested term “{word.surface}”',
            scope="request",
            term=word.surface,
            field_name=term_field,
            expected=1,
            actual=count,
            suggestion=(
                "Retry so the selected source sense appears exactly once in "
                "contextual_cards."),
            identity={
                "normalized": normalized,
                "rank": word.rank,
            }))

    for card_index, card in enumerate(additional_cards):
        if (
                not isinstance(card, dict)
                or not isinstance(card.get(term_field), str)):
            continue
        term = card[term_field]
        normalized = _term_identity(term, pipeline)
        if normalized in requested_terms:
            continue
        problems.append(process_text._generated_problem(
            "unexpected_additional_sense_term",
            "Additional-sense card term was not requested",
            "additional_sense_cards may contain senses only for terms in "
            "this source chunk.",
            path=(
                f"$.{additional_key}[{card_index}].{term_field}"),
            location=(
                f"Response → {additional_key} → item "
                f"{card_index + 1} → {term_field}"),
            scope="field",
            card_index=len(contextual_cards) + card_index,
            term=term,
            field_name=term_field,
            expected=[
                requested_terms[item].surface
                for item in requested_order
            ],
            actual=term,
            suggestion=(
                "Retry and remove every unrequested additional-sense card."),
            identity=normalized))

    return contextual_cards, additional_cards, problems


def _collapse_sentence_arrays(cards):
    """Canonicalize v5 arrays to the pipe fields used by Anki templates."""
    problems = []
    for card_index, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        for field_name in (
                "Sentences",
                _SENTENCE_TRANSLATIONS_FIELD):
            if field_name not in card:
                continue
            value = card[field_name]
            field_path = process_text._field_path(card_index, field_name)
            if not isinstance(value, list):
                problems.append(process_text._generated_problem(
                    "sentence_collection_not_array",
                    "Sentence collection is not an array",
                    "Current source responses must return generated sentences "
                    "and their translations as JSON arrays.",
                    path=field_path,
                    location=f"Card {card_index + 1} → {field_name}",
                    scope="field",
                    card_index=card_index,
                    field_name=field_name,
                    expected="An array containing exactly three strings.",
                    actual=type(value).__name__,
                    suggestion=(
                        "Retry using the frozen structured-output schema."),
                    identity=field_name))
                continue
            if len(value) != 3:
                problems.append(process_text._generated_problem(
                    "sentence_collection_wrong_length",
                    "Sentence array length is wrong",
                    "Generated-example sentence arrays and translation arrays "
                    "must each contain exactly three items.",
                    path=field_path,
                    location=f"Card {card_index + 1} → {field_name}",
                    scope="field",
                    card_index=card_index,
                    field_name=field_name,
                    expected=3,
                    actual=len(value),
                    suggestion=(
                        "Retry using the frozen structured-output schema."),
                    identity=field_name))
            if not all(isinstance(item, str) for item in value):
                problems.append(process_text._generated_problem(
                    "sentence_collection_item_not_text",
                    "Sentence array contains a non-text item",
                    "Every generated sentence and sentence translation must "
                    "be a JSON string.",
                    path=field_path,
                    location=f"Card {card_index + 1} → {field_name}",
                    scope="field",
                    card_index=card_index,
                    field_name=field_name,
                    expected="Only string array items.",
                    actual=[
                        type(item).__name__
                        for item in value
                    ],
                    suggestion=(
                        "Retry using the frozen structured-output schema."),
                    identity=field_name,
                    exception_type="TypeError"))
                continue
            for item_index, item in enumerate(value):
                if "|" not in html.unescape(item):
                    continue
                problems.append(process_text._generated_problem(
                    "sentence_collection_item_contains_delimiter",
                    "Sentence array item contains the card delimiter",
                    "A generated sentence or translation array item cannot "
                    "contain | because the Anki field uses it to separate "
                    "the three aligned items.",
                    path=f"{field_path}[{item_index}]",
                    location=(
                        f"Card {card_index + 1} → {field_name} → item "
                        f"{item_index + 1}"),
                    scope="field",
                    card_index=card_index,
                    field_name=field_name,
                    expected="Text without |.",
                    actual=item,
                    suggestion=(
                        "Retry or replace the delimiter with normal "
                        "punctuation."),
                    identity={
                        "field": field_name,
                        "item_index": item_index,
                    }))
            card[field_name] = "|".join(value)
    return problems


def _term_identity(term, pipeline):
    return normalize_word(term, pipeline.language_key)


def _sense_text_fingerprint(value):
    """Ignore superficial punctuation and case when comparing sense text."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def _translation_definition_pairs(pipeline):
    translations = {}
    definitions = {}
    for setting in pipeline_store.get_requested_field_settings(pipeline):
        field_name = pipeline_store.response_field_name(setting)
        if setting.field_key == "translation":
            translations[setting.target_language_key] = field_name
        elif setting.field_key == "dictionary_meaning":
            definitions[setting.target_language_key] = field_name
    return tuple(
        (
            language_key,
            translations[language_key],
            definitions[language_key],
        )
        for language_key in sorted(
            set(translations) & set(definitions))
    )


def _refresh_report_counts(report):
    problems = report["problems"]
    non_overrideable_count = sum(
        not problem["overrideable"]
        for problem in problems)
    report.update({
        "valid": not problems,
        "problem_count": len(problems),
        "overrideable_problem_count": (
            len(problems) - non_overrideable_count),
        "non_overrideable_problem_count": non_overrideable_count,
        "can_manually_accept": (
            report["structurally_valid"]
            and non_overrideable_count == 0),
    })
    return report


def inspect_pipeline_response(
        raw_text,
        pipeline,
        chunk,
        *,
        use_source_for_example_sentences=False,
        require_sentence_translations=True,
        sentence_collections_as_arrays=False,
        use_source_context_translation_map=False,
        use_split_source_context_cards=False,
        use_grouped_source_results=False,
        use_compact_source_results=False,
        use_local_example_emphasis=False,
        source_context_translation_memory=None,
        include_source_context_nuance=False,
        require_generated_examples=True):
    """Return canonical cards plus precise structural/content problems.

    The returned ``canonical_response`` is intentionally internal data and
    should be removed before showing or persisting the report. It exists only
    when the response is structurally safe to package.
    """
    pipeline_store.validate_pipelines((pipeline,))
    if use_grouped_source_results and use_compact_source_results:
        raise ValueError(
            "Grouped v8 and compact source results are mutually exclusive.")
    if use_local_example_emphasis and not use_compact_source_results:
        raise ValueError(
            "Local example emphasis requires compact source results.")
    use_ranked_source_results = (
        use_grouped_source_results
        or use_compact_source_results)
    if use_ranked_source_results:
        # V8 rank-keyed objects and compact arrays normalize to the same
        # source-context card stream. Keep either public flag self-contained.
        use_source_for_example_sentences = True
        require_sentence_translations = True
        sentence_collections_as_arrays = True
        use_source_context_translation_map = True
        use_split_source_context_cards = True
    local_repair = None
    effective_raw_text = raw_text
    if use_local_example_emphasis:
        local_repair = repair_compact_response(
            raw_text,
            pipeline,
            chunk,
            source_lexical_only=(
                use_source_for_example_sentences
                and not require_generated_examples),
            include_generated_examples=(
                require_generated_examples),
            include_source_context_nuance=(
                include_source_context_nuance))
        effective_raw_text = local_repair.candidate_raw_text
    validation_text = effective_raw_text
    additional_senses_missing_sentences = set()
    preflight_problems = []
    source_context_translations = {}
    source_context_nuances = {}
    grouped_card_raw_paths = []
    compact_card_source_ranks = []
    compact_card_context_ids = []
    split_contextual_card_indices = set()
    split_additional_card_indices = set()
    parsed = None
    parse_failed = False
    if (
            use_source_for_example_sentences
            or sentence_collections_as_arrays
            or use_source_context_translation_map
            or use_split_source_context_cards):
        # Remove any model-supplied value from each term's first entry before
        # ordinary validation. The retained context is authoritative and is
        # inserted below. Later entries remain untouched and are checked.
        try:
            parsed = json.loads(effective_raw_text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
            parse_failed = True
        merged_split_cards = None
        if use_compact_source_results and not parse_failed:
            (
                merged_split_cards,
                grouped_card_raw_paths,
                compact_card_source_ranks,
                compact_card_context_ids,
                source_context_translations,
                source_context_nuances,
                split_contextual_card_indices,
                split_additional_card_indices,
                compact_problems,
            ) = _normalize_compact_source_results(
                parsed,
                pipeline,
                chunk,
                enforce_contextual_constants=(
                    not use_local_example_emphasis),
                source_context_translation_memory=(
                    source_context_translation_memory),
                include_source_context_nuance=(
                    include_source_context_nuance))
            preflight_problems.extend(compact_problems)
        elif use_grouped_source_results and not parse_failed:
            (
                merged_split_cards,
                grouped_card_raw_paths,
                source_context_translations,
                source_context_nuances,
                split_contextual_card_indices,
                split_additional_card_indices,
                grouped_problems,
            ) = _normalize_grouped_source_results(
                parsed,
                pipeline,
                chunk,
                include_source_context_nuance=(
                    include_source_context_nuance))
            preflight_problems.extend(grouped_problems)
        elif use_split_source_context_cards and not parse_failed:
            (
                contextual_cards,
                additional_cards,
                split_problems,
            ) = _split_source_card_collections(
                parsed,
                pipeline,
                chunk)
            preflight_problems.extend(split_problems)
            merged_split_cards = [
                *contextual_cards,
                *additional_cards,
            ]
            split_contextual_card_indices = set(
                range(len(contextual_cards)))
            split_additional_card_indices = set(
                range(
                    len(contextual_cards),
                    len(merged_split_cards)))
        if (
                use_source_context_translation_map
                and not use_ranked_source_results
                and not parse_failed):
            (
                source_context_translations,
                map_problems,
            ) = _source_context_translation_map(parsed, chunk)
            preflight_problems.extend(map_problems)
        if (
                (use_split_source_context_cards
                 or use_ranked_source_results)
                and not parse_failed):
            parsed = {"cards": merged_split_cards}
        cards = (
            parsed.get("cards")
            if isinstance(parsed, dict)
            else parsed)
        if isinstance(cards, list):
            if sentence_collections_as_arrays:
                preflight_problems.extend(
                    _collapse_sentence_arrays(cards))
            requested = {
                _term_identity(word.surface, pipeline)
                for word in chunk.words
            }
            seen = set()
            term_field = pipeline_store.get_language(
                pipeline.language_key).term_field
            words_by_term = {
                _term_identity(word.surface, pipeline): word
                for word in chunk.words
            }
            for card_index, card in enumerate(cards):
                if not isinstance(card, dict):
                    continue
                term = card.get(term_field)
                if not isinstance(term, str):
                    continue
                normalized = _term_identity(term, pipeline)
                if normalized not in requested:
                    continue
                is_contextual_card = (
                    card_index in split_contextual_card_indices
                    if (
                        use_ranked_source_results
                        or use_split_source_context_cards)
                    else normalized not in seen)
                if is_contextual_card:
                    seen.add(normalized)
                    if use_source_for_example_sentences:
                        if (
                                use_source_context_translation_map
                                and "Sentences" in card):
                            preflight_problems.append(
                                process_text._generated_problem(
                                    "contextual_card_contains_sentences",
                                    (
                                        "Contextual card repeats its source "
                                        "example"),
                                    (
                                        "The first card for each term must "
                                        "omit Sentences; the processor inserts "
                                        "the exact audited source occurrence."),
                                    path=process_text._field_path(
                                        card_index,
                                        "Sentences"),
                                    location=(
                                        f"Card {card_index + 1} → Sentences"),
                                    scope="field",
                                    card_index=card_index,
                                    term=term,
                                    field_name="Sentences",
                                    expected="Field omitted",
                                    actual="Field present",
                                    suggestion=(
                                        "Retry using the frozen structured-"
                                        "output schema.")))
                        card.pop("Sentences", None)
                    if use_source_context_translation_map:
                        if _SENTENCE_TRANSLATIONS_FIELD in card:
                            preflight_problems.append(
                                process_text._generated_problem(
                                    "contextual_card_contains_translation",
                                    (
                                        "Contextual card repeats its source "
                                        "translation"),
                                    (
                                        "The first card for each term must "
                                        "omit Sentence Translations (English); "
                                        "the processor injects the translation "
                                        "matched by context_id."),
                                    path=process_text._field_path(
                                        card_index,
                                        _SENTENCE_TRANSLATIONS_FIELD),
                                    location=(
                                        f"Card {card_index + 1} → "
                                        + _SENTENCE_TRANSLATIONS_FIELD),
                                    scope="field",
                                    card_index=card_index,
                                    term=term,
                                    field_name=(
                                        _SENTENCE_TRANSLATIONS_FIELD),
                                    expected="Field omitted",
                                    actual="Field present",
                                    suggestion=(
                                        "Retry using the frozen structured-"
                                        "output schema.")))
                            card.pop(
                                _SENTENCE_TRANSLATIONS_FIELD,
                                None)
                        word = words_by_term.get(normalized)
                        if word is not None:
                            translation = source_context_translations.get(
                                word.context_id)
                            if use_ranked_source_results:
                                # Root translations are validated exactly once
                                # before generic card validation.  A safe
                                # placeholder prevents one malformed shared
                                # value from creating duplicate per-card
                                # diagnostics; the real value is restored to
                                # canonical cards below.
                                card[_SENTENCE_TRANSLATIONS_FIELD] = (
                                    _GROUPED_CONTEXT_TRANSLATION_PLACEHOLDER)
                            elif translation is not None:
                                card[_SENTENCE_TRANSLATIONS_FIELD] = (
                                    translation)
                elif (
                        (
                            not use_split_source_context_cards
                            or card_index in split_additional_card_indices)
                        and require_generated_examples
                        and "Sentences" not in card):
                    additional_senses_missing_sentences.add(card_index)
            validation_text = json.dumps(
                parsed,
                ensure_ascii=False)
    optional_fields = {"Sentences"}
    if use_source_for_example_sentences and not require_generated_examples:
        optional_fields.add(_SENTENCE_TRANSLATIONS_FIELD)
        lexical_source_fields = {
            pipeline_store.response_field_name(field_setting)
            for field_setting
            in pipeline_store.get_source_lexical_field_settings(pipeline)
        }
        optional_fields.update(
            pipeline_store.response_field_name(field_setting)
            for field_setting
            in pipeline_store.get_requested_field_settings(pipeline)
            if (
                pipeline_store.response_field_name(field_setting)
                not in lexical_source_fields))
    report = process_text.inspect_generated_response(
        validation_text,
        pipeline,
        optional_fields=(
            tuple(sorted(optional_fields))
            if use_source_for_example_sentences
            else ()),
        require_sentence_translations=require_sentence_translations)
    if local_repair is not None:
        audit = local_repair.audit_record()
        report.update({
            "local_repair_applied": local_repair.changed,
            "local_repair_count": len(local_repair.changes),
            "local_repairs": list(local_repair.changes),
            "_local_repair_audit": audit,
            "_effective_raw_text": effective_raw_text,
        })
    if preflight_problems:
        report["problems"] = [
            *preflight_problems,
            *report["problems"],
        ]
        report["structurally_valid"] = False
        _refresh_report_counts(report)
    canonical = report.get("canonical_response")
    if canonical is None:
        if use_compact_source_results:
            _finalize_compact_report(
                report,
                grouped_card_raw_paths,
                compact_card_source_ranks,
                compact_card_context_ids)
        elif use_grouped_source_results:
            _remap_grouped_card_problem_paths(
                report,
                grouped_card_raw_paths)
        return report

    term_field = pipeline_store.get_language(
        pipeline.language_key).term_field
    if use_source_for_example_sentences:
        contexts = {
            context.context_id: context
            for context in chunk.contexts
        }
        words_by_term = {
            _term_identity(word.surface, pipeline): word
            for word in chunk.words
        }
        seen = set()
        for card_index, card in enumerate(canonical["cards"]):
            term = card[term_field]
            normalized = _term_identity(term, pipeline)
            word = words_by_term.get(normalized)
            if word is None:
                continue
            is_contextual_card = (
                card_index in split_contextual_card_indices
                if (
                    use_ranked_source_results
                    or use_split_source_context_cards)
                else normalized not in seen)
            if is_contextual_card:
                seen.add(normalized)
                if use_ranked_source_results:
                    translation = source_context_translations.get(
                        word.context_id)
                    card[_SENTENCE_TRANSLATIONS_FIELD] = (
                        translation
                        if isinstance(translation, str)
                        else "")
                context = contexts.get(word.context_id)
                if context is None:
                    report["problems"].append(
                        process_text._generated_problem(
                            "missing_source_example_context",
                            "Source example context is unavailable",
                            "The contextual sense cannot receive its exact "
                            "source example because the requested context is "
                            "missing.",
                            path=(
                                f"request.words[rank={word.rank}]"
                                ".context_id"),
                            location=f'Requested term “{word.surface}”',
                            scope="request",
                            overrideable=False,
                            term=word.surface,
                            field_name="Sentences",
                            expected=(
                                "A retained source context matching the "
                                "word's context_id."),
                            actual=word.context_id,
                            suggestion=(
                                "Retry with source context enabled or create "
                                "a new source job.")))
                else:
                    context_text = context.text
                    start_offset = getattr(word, "start_offset", None)
                    end_offset = getattr(word, "end_offset", None)
                    if start_offset is not None and end_offset is not None:
                        relative_start = (
                            start_offset - context.start_offset)
                        relative_end = (
                            end_offset - context.start_offset)
                        try:
                            emphasized_context = (
                                process_text.emphasize_source_occurrence(
                                    context_text,
                                    word.surface,
                                    pipeline.language_key,
                                    start_offset=relative_start,
                                    end_offset=relative_end))
                        except ValueError as error:
                            report["problems"].append(
                                process_text._generated_problem(
                                    "invalid_source_occurrence_span",
                                    "Source occurrence span is inconsistent",
                                    "The audited occurrence span no longer "
                                    "identifies this term inside its retained "
                                    "context.",
                                    path=(
                                        f"request.words[rank={word.rank}]"
                                        ".occurrence_span"),
                                    location=(
                                        f'Requested term “{word.surface}”'),
                                    scope="request",
                                    overrideable=False,
                                    term=word.surface,
                                    field_name="Sentences",
                                    expected=word.surface,
                                    actual={
                                        "occurrence_span": [
                                            relative_start,
                                            relative_end,
                                        ],
                                        "error": str(error),
                                    },
                                    suggestion=(
                                        "Create a new source job from the "
                                        "current audited corpus build.")))
                            emphasized_context = (
                                process_text.emphasize_term_in_sentences(
                                    context_text,
                                    "",
                                    pipeline.language_key))
                    elif use_source_context_translation_map:
                        report["problems"].append(
                            process_text._generated_problem(
                                "missing_source_occurrence_span",
                                "Source occurrence span is unavailable",
                                "Current source-context requests require both "
                                "audited offsets so only the selected "
                                "occurrence is highlighted.",
                                path=(
                                    f"request.words[rank={word.rank}]"
                                    ".occurrence_span"),
                                location=(
                                    f'Requested term “{word.surface}”'),
                                scope="request",
                                overrideable=False,
                                term=word.surface,
                                field_name="Sentences",
                                expected=(
                                    "Both start_offset and end_offset for the "
                                    "audited source occurrence."),
                                actual={
                                    "start_offset": start_offset,
                                    "end_offset": end_offset,
                                },
                                suggestion=(
                                    "Create a new source job from the current "
                                    "audited corpus build.")))
                        emphasized_context = (
                            process_text.emphasize_term_in_sentences(
                                context_text,
                                "",
                                pipeline.language_key))
                    else:
                        emphasized_context = (
                            process_text.emphasize_term_in_sentences(
                                context_text,
                                word.surface,
                                pipeline.language_key))
                    card["Sentences"] = (
                        process_text.encode_source_context_block(
                            emphasized_context))
                    if (
                            require_sentence_translations
                            and not use_ranked_source_results):
                        translations = card.get(
                            _SENTENCE_TRANSLATIONS_FIELD,
                            "")
                        if not translations.strip():
                            report["problems"].append(
                                process_text._generated_problem(
                                    "blank_source_context_translation",
                                    "Source-context translation is blank",
                                    "Every retained source context needs one "
                                    "complete, non-blank English translation.",
                                    path=process_text._field_path(
                                        card_index,
                                        _SENTENCE_TRANSLATIONS_FIELD),
                                    location=(
                                        f'Card {card_index + 1} (“{term}”) → '
                                        + _SENTENCE_TRANSLATIONS_FIELD),
                                    scope="field",
                                    overrideable=False,
                                    card_index=card_index,
                                    term=term,
                                    field_name=(
                                        _SENTENCE_TRANSLATIONS_FIELD),
                                    expected=(
                                        "One complete English translation."),
                                    actual="Blank text",
                                    suggestion=(
                                        "Retry so the retained context is "
                                        "translated before packaging.")))
                        translation_count = len(translations.split("|"))
                        sentence_count = 1
                        if translation_count != sentence_count:
                            report["problems"].append(
                                process_text._generated_problem(
                                    "sentence_translation_count_mismatch",
                                    (
                                        "Source-context translation does not "
                                        "match the retained example"),
                                    (
                                        "The contextual sense uses the exact "
                                        "retained source passage. It needs one "
                                        "complete English translation for that "
                                        "single passage."),
                                    path=process_text._field_path(
                                        card_index,
                                        _SENTENCE_TRANSLATIONS_FIELD),
                                    location=(
                                        f'Card {card_index + 1} (“{term}”) → '
                                        + _SENTENCE_TRANSLATIONS_FIELD),
                                    scope="field",
                                    overrideable=False,
                                    card_index=card_index,
                                    term=term,
                                    field_name=(
                                        _SENTENCE_TRANSLATIONS_FIELD),
                                    expected={
                                        "translation_count": sentence_count,
                                    },
                                    actual={
                                        "translation_count": (
                                            translation_count),
                                    },
                                    suggestion=(
                                        "Provide one complete English "
                                        "translation of the exact retained "
                                        "source context, without using | "
                                        "inside it.")))
                        if _HTML_TAG_PATTERN.search(translations):
                            report["problems"].append(
                                process_text._generated_problem(
                                    "source_context_translation_contains_html",
                                    "Source-context translation contains HTML",
                                    "A retained context translation must be "
                                    "plain English text without HTML.",
                                    path=process_text._field_path(
                                        card_index,
                                        _SENTENCE_TRANSLATIONS_FIELD),
                                    location=(
                                        f'Card {card_index + 1} (“{term}”) → '
                                        + _SENTENCE_TRANSLATIONS_FIELD),
                                    scope="field",
                                    overrideable=False,
                                    card_index=card_index,
                                    term=term,
                                    field_name=(
                                        _SENTENCE_TRANSLATIONS_FIELD),
                                    expected="Plain text without HTML.",
                                    actual=translations,
                                    suggestion=(
                                        "Retry or remove the HTML tags after "
                                        "checking the translation.")))
                continue
            if (
                    require_generated_examples
                    and card_index in additional_senses_missing_sentences):
                report["problems"].append(
                    process_text._generated_problem(
                        "missing_additional_sense_sentences",
                        "Additional sense has no example sentences",
                        "Only the first, contextual sense may omit "
                        '"Sentences"; every additional sense needs three '
                        "generated examples.",
                        path=process_text._field_path(
                            card_index,
                            "Sentences"),
                        location=(
                            f'Card {card_index + 1} (“{term}”) '
                            "→ Sentences"),
                        scope="field",
                        overrideable=False,
                        card_index=card_index,
                        term=term,
                        field_name="Sentences",
                        expected=(
                            "Three distinct examples for this additional "
                            "sense."),
                        actual="Field omitted",
                        suggestion=(
                            "Retry the request or add three example sentences "
                            "before packaging.")))

        generated_examples = {}
        legacy_seen_terms = set()
        for card_index, card in enumerate(canonical["cards"]):
            term = card[term_field]
            normalized = _term_identity(term, pipeline)
            if (
                    use_ranked_source_results
                    or use_split_source_context_cards):
                is_additional = (
                    card_index in split_additional_card_indices)
            else:
                is_additional = normalized in legacy_seen_terms
                legacy_seen_terms.add(normalized)
            if not is_additional:
                continue
            sentences = card.get("Sentences")
            if not isinstance(sentences, str):
                continue
            if use_local_example_emphasis:
                if pipeline.language_key.startswith(
                        "classical_chinese"):
                    for sentence_index, sentence in enumerate(
                            sentences.split("|")):
                        if term in html.unescape(sentence):
                            continue
                        report["problems"].append(
                            process_text._generated_problem(
                                "missing_exact_form_source_term",
                                (
                                    "Example omits the complete source "
                                    "term"),
                                (
                                    "Classical Chinese does not require "
                                    "inflection here, so every generated "
                                    "additional-sense example must contain "
                                    "the complete requested lexical item at "
                                    "least once."),
                                path=(
                                    process_text._field_path(
                                        card_index,
                                        "Sentences")
                                    + f"[{sentence_index}]"),
                                location=(
                                    f'Card {card_index + 1} (“{term}”) '
                                    "→ Sentences → item "
                                    f"{sentence_index + 1}"),
                                scope="field",
                                card_index=card_index,
                                term=term,
                                field_name="Sentences",
                                expected=(
                                    "At least one literal occurrence of "
                                    f"{term}."),
                                actual=sentence,
                                suggestion=(
                                    "Retry with a natural example that uses "
                                    "the complete requested term, not only a "
                                    "component or synonym."),
                                identity={
                                    "sentence_index": sentence_index,
                                    "term": term,
                                }))
                # V10 owns presentation markup locally. Exact literal matches
                # are emphasized using the language's boundary policy.
                # Inflected forms may remain unmarked; the card template also
                # displays the immutable term.
                sentences = process_text.emphasize_term_in_sentences(
                    sentences,
                    term,
                    pipeline.language_key)
                card["Sentences"] = sentences
            example_set = tuple(sorted(
                sentence.strip()
                for sentence in sentences.split("|")))
            previous = generated_examples.get(
                (normalized, example_set))
            if previous is None:
                generated_examples[(normalized, example_set)] = card_index
                continue
            report["problems"].append(
                process_text._generated_problem(
                    "duplicate_additional_sense_examples",
                    "Additional senses reuse the same examples",
                    "Different additional senses of one term must use "
                    "different generated example sets so each card "
                    "demonstrates its own meaning.",
                    path=process_text._field_path(
                        card_index,
                        "Sentences"),
                    location=(
                        f'Card {card_index + 1} (“{term}”) → Sentences'),
                    scope="field",
                    overrideable=False,
                    card_index=card_index,
                    term=term,
                    field_name="Sentences",
                    expected=(
                        "Three examples distinct from every other additional "
                        "sense of this term."),
                    actual={
                        "duplicates_card": previous + 1,
                        "sentences": list(example_set),
                    },
                    suggestion=(
                        "Retry with sense-specific examples for each "
                        "additional card."),
                    identity={
                        "normalized": normalized,
                        "first_card_index": previous,
                    }))

        if use_ranked_source_results:
            # Dedicated source-sentence notes own contextual examples.
            # Additional senses retain their three generated examples.
            for card_index in split_contextual_card_indices:
                if not 0 <= card_index < len(canonical["cards"]):
                    continue
                canonical["cards"][card_index].pop("Sentences", None)
                canonical["cards"][card_index].pop(
                    _SENTENCE_TRANSLATIONS_FIELD,
                    None)

        words_by_rank = {
            word.rank: word
            for word in chunk.words
        }
        word_ranks_by_context = {}
        for word in chunk.words:
            word_ranks_by_context.setdefault(
                getattr(word, "context_id", None),
                []).append(word.rank)
        source_context_records = []
        for context in chunk.contexts:
            context_id = context.context_id
            sentence_ids = tuple(
                getattr(context, "sentence_ids", ()))
            if (
                    use_ranked_source_results
                    and not require_generated_examples
                    and len(sentence_ids) != 1):
                report["problems"].append(
                    process_text._generated_problem(
                        "source_context_is_not_one_sentence",
                        "Source sentence context is not exactly one sentence",
                        "Shared Sentence → Meaning notes require each retained "
                        "context to identify exactly one original source "
                        "sentence.",
                        path="$",
                        location=f"Source context {context_id}",
                        expected="Exactly one sentence_id.",
                        actual={
                            "sentence_ids": list(sentence_ids),
                        },
                        suggestion=(
                            "Rebuild this request using Current sentence "
                            "context."),
                        identity=context_id))
            context_word_ranks = tuple(
                getattr(
                    context,
                    "word_ranks",
                    word_ranks_by_context.get(context_id, ())))
            source_context_records.append({
                "context_id": context.context_id,
                "sentence_id": (
                    sentence_ids[0]
                    if len(sentence_ids) == 1
                    else context.context_id),
                "original_sentence": getattr(context, "text", ""),
                "english_translation": source_context_translations.get(
                    context.context_id,
                    ""),
                "nuance": source_context_nuances.get(
                    context.context_id,
                    ""),
                "word_ranks": list(context_word_ranks),
                "terms": [
                    words_by_rank[rank].surface
                    for rank in context_word_ranks
                    if rank in words_by_rank
                ],
                "ranked_terms": [
                    {
                        "rank": rank,
                        "term": words_by_rank[rank].surface,
                    }
                    for rank in context_word_ranks
                    if rank in words_by_rank
                ],
            })
        canonical["source_contexts"] = source_context_records

        sense_pairs = _translation_definition_pairs(pipeline)
        lexical_field_names = tuple(
            field_name
            for field_name in process_text.get_response_field_names(
                pipeline,
                include_sentence_translations=True)
            if field_name not in {
                term_field,
                "Sentences",
                _SENTENCE_TRANSLATIONS_FIELD,
            })
        seen_senses = {}
        for card_index, card in enumerate(canonical["cards"]):
            term = card[term_field]
            normalized = _term_identity(term, pipeline)
            fingerprints = tuple(
                (
                    language_key,
                    _sense_text_fingerprint(card.get(translation_field)),
                    _sense_text_fingerprint(card.get(definition_field)),
                )
                for (
                    language_key,
                    translation_field,
                    definition_field,
                ) in sense_pairs
            )
            fingerprints = tuple(
                fingerprint
                for fingerprint in fingerprints
                if fingerprint[1] and fingerprint[2])
            if not fingerprints:
                continue
            identity = (normalized, fingerprints)
            if use_local_example_emphasis:
                # V10 may receive homographs/polyphones whose concise
                # translation and definition coincide while pronunciation,
                # part of speech, register, or nuance distinguishes the
                # lexical sense. Never collapse those locally.
                identity = (
                    *identity,
                    tuple(
                        (
                            field_name,
                            _sense_text_fingerprint(
                                card.get(field_name)),
                        )
                        for field_name in lexical_field_names),
                )
            previous = seen_senses.get(identity)
            if previous is None:
                seen_senses[identity] = card_index
                continue
            report["problems"].append(
                process_text._generated_problem(
                    "duplicate_source_sense",
                    "A term repeats the same lexical sense",
                    "Contextual and additional cards for one term must "
                    "represent genuinely disjoint senses. This card repeats "
                    "the same translation and dictionary explanation as "
                    "another card.",
                    path=f"$.cards[{card_index}]",
                    location=f'Card {card_index + 1} (“{term}”)',
                    scope="card",
                    overrideable=False,
                    card_index=card_index,
                    term=term,
                    expected=(
                        "A translation and dictionary explanation distinct "
                        "from every other sense card for this term."),
                    actual={
                        "duplicates_card": previous + 1,
                        "sense_fields": [
                            {
                                "language_key": language_key,
                                "translation": translation,
                                "dictionary_meaning": definition,
                            }
                            for (
                                language_key,
                                translation,
                                definition,
                            ) in fingerprints
                        ],
                    },
                    suggestion=(
                        "Retry and remove the duplicate sense, keeping the "
                        "source-selected occurrence in contextual_cards."),
                    identity={
                        "normalized": normalized,
                        "fingerprints": fingerprints,
                    }))
    rank_by_term = {
        _term_identity(word.surface, pipeline): word.rank
        for word in chunk.words
    }
    returned_terms = set()
    indexed_cards = list(enumerate(canonical["cards"]))
    for card_index, card in indexed_cards:
        term = card[term_field]
        normalized = _term_identity(term, pipeline)
        returned_terms.add(normalized)
        if normalized in rank_by_term:
            continue
        location = f'Card {card_index + 1} (“{term}”) → {term_field}'
        report["problems"].append(process_text._generated_problem(
            "term_outside_source_chunk",
            "Card term was not requested in this batch",
            "The response contains a term outside this source chunk: "
            f"{term}",
            path=process_text._field_path(card_index, term_field),
            location=location,
            scope="field",
            overrideable=True,
            card_index=card_index,
            term=term,
            field_name=term_field,
            expected=(
                "One of the terms listed in this source request."),
            actual=term,
            suggestion=(
                "Check whether this is a legitimate spelling/form for the "
                "source passage. Retry, edit, or explicitly accept it."),
            identity=normalized))

    for word in chunk.words:
        normalized = _term_identity(word.surface, pipeline)
        if normalized in returned_terms:
            continue
        report["problems"].append(process_text._generated_problem(
            "missing_source_term",
            "Requested source term has no card",
            "The response omitted requested source term: "
            f"{word.surface}",
            path=f"request.words[rank={word.rank}]",
            location=f'Requested term “{word.surface}”',
            scope="request",
            overrideable=True,
            term=word.surface,
            field_name=term_field,
            expected="At least one returned card for this term.",
            actual="No matching card was returned.",
            suggestion=(
                "Retry to generate the omitted card, or explicitly accept "
                "that this batch will not include it."),
            identity={
                "normalized": normalized,
                "rank": word.rank,
            }))

    # Restore requested source order while keeping multiple senses stable.
    # A manually accepted out-of-batch term is placed after every requested
    # term in its original response order instead of causing a KeyError.
    canonical["cards"] = [
        card
        for _original_index, card in sorted(
            indexed_cards,
            key=lambda pair: (
                0,
                rank_by_term[
                    _term_identity(pair[1][term_field], pipeline)],
                pair[0],
            )
            if _term_identity(
                pair[1][term_field],
                pipeline) in rank_by_term
            else (1, pair[0], pair[0]))
    ]
    if use_compact_source_results:
        _finalize_compact_report(
            report,
            grouped_card_raw_paths,
            compact_card_source_ranks,
            compact_card_context_ids)
    elif use_grouped_source_results:
        _remap_grouped_card_problem_paths(
            report,
            grouped_card_raw_paths)
    return _refresh_report_counts(report)


def public_validation_report(report, accepted_problem_ids=()):
    """Return a GUI/persistence-safe report annotated with review state."""
    accepted = frozenset(accepted_problem_ids)
    problems = []
    for problem in report["problems"]:
        value = dict(problem)
        value["accepted"] = value["problem_id"] in accepted
        problems.append(value)
    remaining = [
        problem
        for problem in problems
        if not problem["accepted"]
    ]
    result = {
        key: value
        for key, value in report.items()
        if (
            key not in {"canonical_response", "problems"}
            and not key.startswith("_"))
    }
    result.update({
        "problems": problems,
        "accepted_problem_count": (
            len(problems) - len(remaining)),
        "remaining_problem_count": len(remaining),
        "all_problems_resolved": not remaining,
        "can_complete_with_manual_acceptance": (
            report["structurally_valid"]
            and all(
                problem["overrideable"]
                for problem in remaining)),
    })
    return result


def _validation_error(report):
    first = report["problems"][0]
    exception_class = (
        TypeError
        if first.get("exception_type") == "TypeError"
        else process_text.GeneratedCardValidationError)
    error = exception_class(first["message"])
    error.validation_report = public_validation_report(report)
    error.local_repair_audit = report.get(
        "_local_repair_audit")
    error.effective_raw_text = report.get(
        "_effective_raw_text")
    return error


def make_pipeline_response_validator(
        pipeline,
        *,
        use_source_for_example_sentences=False,
        require_sentence_translations=True,
        sentence_collections_as_arrays=False,
        use_source_context_translation_map=False,
        use_split_source_context_cards=False,
        use_grouped_source_results=False,
        use_compact_source_results=False,
        use_local_example_emphasis=False,
        source_context_translation_memory_by_chunk=None,
        include_source_context_nuance=False,
        require_generated_examples=True):
    """Validate schema, card semantics, and membership in the source chunk."""
    pipeline_store.validate_pipelines((pipeline,))
    if source_context_translation_memory_by_chunk is None:
        source_context_translation_memory_by_chunk = {}
    if not isinstance(source_context_translation_memory_by_chunk, dict):
        raise TypeError(
            "Source-context translation memory by chunk must be an object.")

    def validate(raw_text, chunk):
        expected_context_ids = {
            context.context_id
            for context in chunk.contexts
        }
        chunk_memory = {
            context_id: hit
            for context_id, hit in
            source_context_translation_memory_by_chunk.get(
                getattr(chunk, "chunk_id", None),
                {}).items()
            if context_id in expected_context_ids
        }
        report = inspect_pipeline_response(
            raw_text,
            pipeline,
            chunk,
            use_source_for_example_sentences=(
                use_source_for_example_sentences),
            require_sentence_translations=(
                require_sentence_translations),
            sentence_collections_as_arrays=(
                sentence_collections_as_arrays),
            use_source_context_translation_map=(
                use_source_context_translation_map),
            use_split_source_context_cards=(
                use_split_source_context_cards),
            use_grouped_source_results=(
                use_grouped_source_results),
            use_compact_source_results=(
                use_compact_source_results),
            use_local_example_emphasis=(
                use_local_example_emphasis),
            source_context_translation_memory=(
                chunk_memory),
            include_source_context_nuance=(
                include_source_context_nuance),
            require_generated_examples=(
                require_generated_examples))
        if report["problems"]:
            raise _validation_error(report)
        return ValidatedPipelineResponse(
            report["canonical_response"],
            local_repair_audit=report.get(
                "_local_repair_audit"),
            effective_raw_text=report.get(
                "_effective_raw_text"))

    return validate
