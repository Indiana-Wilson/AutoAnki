"""Conservative, auditable local repairs for compact source responses.

The provider response remains immutable evidence.  This module returns a
separate candidate string containing only transformations that can be derived
from the frozen request or that remove unusable optional output.  It never
invents lexical content, translations, examples, ranks, or context IDs.
"""

from dataclasses import dataclass
import hashlib
import html
from html.parser import HTMLParser
import json
import re

import pipeline_store
import process_text


_BLOCK_HTML_TAGS = frozenset({
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
})
_SAFE_TEXT_HTML_TAGS = _BLOCK_HTML_TAGS | frozenset({
    "a",
    "abbr",
    "b",
    "bdi",
    "bdo",
    "cite",
    "code",
    "dd",
    "del",
    "dl",
    "dt",
    "em",
    "font",
    "i",
    "ins",
    "kbd",
    "mark",
    "q",
    "s",
    "samp",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "time",
    "u",
    "var",
})
_HTML_LIKE_PATTERN = re.compile(
    r"</?[A-Za-z][^>]*>|&#(?:x[0-9A-Fa-f]+|\d+);|&[A-Za-z][A-Za-z0-9]+;")
_FINGERPRINT_PATTERN = re.compile(r"[\W_]+", re.UNICODE)


@dataclass(frozen=True)
class CompactLocalRepairResult:
    """One immutable provider response and its locally repaired candidate."""

    original_raw_text: str
    candidate_raw_text: str
    original_sha256: str
    candidate_sha256: str
    changes: tuple[dict, ...]

    @property
    def changed(self):
        return self.original_raw_text != self.candidate_raw_text

    def audit_record(self):
        return {
            "schema_version": 1,
            "kind": "compact_local_response_repair",
            "changed": self.changed,
            "change_count": len(self.changes),
            "original_sha256": self.original_sha256,
            "candidate_sha256": self.candidate_sha256,
            "changes": list(self.changes),
        }


class _PlainTextHTMLParser(HTMLParser):
    """Preserve visible text while discarding presentation markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.safe = True

    def _boundary(self):
        if self.parts and not self.parts[-1].endswith((" ", "\n", "\t")):
            self.parts.append(" ")

    def handle_starttag(self, tag, _attrs):
        normalized = tag.casefold()
        if normalized not in _SAFE_TEXT_HTML_TAGS:
            self.safe = False
        if normalized in _BLOCK_HTML_TAGS:
            self._boundary()

    def handle_startendtag(self, tag, _attrs):
        normalized = tag.casefold()
        if normalized not in _SAFE_TEXT_HTML_TAGS:
            self.safe = False
        if normalized in _BLOCK_HTML_TAGS:
            self._boundary()

    def handle_endtag(self, tag):
        normalized = tag.casefold()
        if normalized not in _SAFE_TEXT_HTML_TAGS:
            self.safe = False
        if normalized in _BLOCK_HTML_TAGS:
            self._boundary()

    def handle_data(self, data):
        self.parts.append(data)

    def text(self):
        return "".join(self.parts).strip()


def _sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded(value, limit=160):
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [{len(text) - limit:,} more characters]"


def _change(rule_id, path, action, **details):
    return {
        "rule_id": rule_id,
        "path": path,
        "action": action,
        **{
            key: _bounded(value)
            for key, value in details.items()
            if value is not None
        },
    }


def _plain_text(value):
    """Return visible text, or the original string when it has no markup."""
    if not isinstance(value, str) or not _HTML_LIKE_PATTERN.search(value):
        return value
    parser = _PlainTextHTMLParser()
    try:
        parser.feed(value)
        parser.close()
    except (ValueError, AssertionError):
        return value
    if not parser.safe:
        return value
    return parser.text()


def _fingerprint(value):
    if not isinstance(value, str):
        return ""
    return _FINGERPRINT_PATTERN.sub(
        " ",
        html.unescape(value).casefold()).strip()


def _sense_fingerprint(sense, lexical_fields):
    """Compare every requested lexical field before discarding a sense."""
    if not isinstance(sense, dict):
        return ()
    return tuple(
        (
            field_name,
            _fingerprint(sense.get(field_name)),
        )
        for field_name in sorted(lexical_fields)
    )


def _strict_substrings(value):
    return {
        value[start:end]
        for start in range(len(value))
        for end in range(start + 1, len(value) + 1)
        if end - start < len(value)
    }


def _component_only_sense(sense, term, language_key):
    """Recognize an unusable whole-term sense without interpreting meaning.

    Classical Chinese is an exact-form language in the current source
    pipelines. If a multi-character term is absent from nearly all examples
    while one strict component appears in every example, the optional object
    is targeting that component rather than the requested complete term. This
    rule deliberately does not run for languages with productive inflection.
    """
    if (
            not language_key.startswith("classical_chinese")
            or not isinstance(term, str)
            or len(term) < 2
            or not isinstance(sense, dict)):
        return None
    sentences = sense.get("Sentences")
    if not isinstance(sentences, list) or not sentences:
        return None
    plain_sentences = [
        _plain_text(sentence)
        for sentence in sentences
    ]
    if any(
            not isinstance(sentence, str) or not sentence.strip()
            for sentence in plain_sentences):
        return None
    full_term_count = sum(
        term in sentence
        for sentence in plain_sentences)
    # One incidental whole-term occurrence does not rescue an object whose
    # examples consistently target a component. Requiring the full term in
    # more than a quarter of examples keeps this pruning rule deliberately
    # narrower than ordinary exact-form validation.
    if full_term_count > max(1, len(plain_sentences) // 4):
        return None
    common_components = {
        component
        for component in _strict_substrings(term)
        if all(component in sentence for sentence in plain_sentences)
    }
    if not common_components:
        return None
    return sorted(
        common_components,
        key=lambda item: (-len(item), item))[0]


def _complete_term_absent_from_sense(sense, term, language_key):
    """Recognize a well-shaped optional Chinese sense with no target usage."""
    if (
            not language_key.startswith("classical_chinese")
            or not isinstance(term, str)
            or not term
            or not isinstance(sense, dict)):
        return False
    sentences = sense.get("Sentences")
    translations = sense.get(
        process_text.SENTENCE_TRANSLATIONS_FIELD_NAME)
    if (
            not isinstance(sentences, list)
            or len(sentences) != 3
            or not isinstance(translations, list)
            or len(translations) != 3):
        return False
    plain_sentences = []
    for value in sentences:
        plain = _plain_text(value)
        if (
                not isinstance(plain, str)
                or not plain.strip()
                or "|" in html.unescape(plain)
                or (
                    plain == value
                    and isinstance(value, str)
                    and _HTML_LIKE_PATTERN.search(value))):
            return False
        plain_sentences.append(plain)
    if any(
            not isinstance(value, str)
            or not value.strip()
            or "|" in html.unescape(value)
            for value in translations):
        return False
    return not any(
        term in sentence
        for sentence in plain_sentences)


def _normalise_identity_array(
        entries,
        *,
        identity_key,
        expected_order,
        path,
        changes):
    """Remove provably irrelevant/identical entries and restore saved order."""
    if not isinstance(entries, list):
        return entries
    expected = set(expected_order)
    kept = []
    seen = {}
    conflicting = set()
    for index, entry in enumerate(entries):
        item_path = f"{path}[{index}]"
        if not isinstance(entry, dict):
            changes.append(_change(
                "drop_unaddressable_item",
                item_path,
                "removed item that cannot represent a requested identity"))
            continue
        identity = entry.get(identity_key)
        try:
            known = identity in expected
        except TypeError:
            known = False
        if not known:
            changes.append(_change(
                "drop_unrequested_identity",
                item_path,
                "removed item outside the frozen request",
                identity=identity))
            continue
        previous = seen.get(identity)
        if previous is None:
            seen[identity] = entry
            kept.append(entry)
            continue
        if previous == entry:
            changes.append(_change(
                "deduplicate_identical_identity",
                item_path,
                "removed byte-equivalent duplicate",
                identity=identity))
            continue
        conflicting.add(identity)
        kept.append(entry)

    identities = [
        entry.get(identity_key)
        for entry in kept
    ]
    if (
            not conflicting
            and len(identities) == len(expected_order)
            and set(identities) == expected):
        order = {
            identity: index
            for index, identity in enumerate(expected_order)
        }
        sorted_entries = sorted(
            kept,
            key=lambda entry: order[entry[identity_key]])
        if sorted_entries != kept:
            changes.append(_change(
                "restore_frozen_identity_order",
                path,
                "sorted items by immutable request order"))
        return sorted_entries
    return kept


def _remove_extra_fields(value, allowed, path, changes):
    if not isinstance(value, dict):
        return
    extras = tuple(key for key in value if key not in allowed)
    for key in extras:
        del value[key]
        changes.append(_change(
            "remove_unrequested_field",
            f"{path}[{json.dumps(key, ensure_ascii=False)}]",
            "removed field outside the strict response contract",
            field=key))


def _remove_html_from_string_fields(
        value,
        field_names,
        path,
        changes,
        **details):
    """Strip only safe presentation markup from selected string fields."""
    if not isinstance(value, dict):
        return
    for field_name in sorted(field_names):
        field_value = value.get(field_name)
        plain = _plain_text(field_value)
        if plain == field_value:
            continue
        value[field_name] = plain
        changes.append(_change(
            "remove_model_html",
            f"{path}[{json.dumps(field_name, ensure_ascii=False)}]",
            "preserved visible text and removed provider HTML",
            **details))


def repair_compact_response(
        raw_text,
        pipeline,
        chunk,
        *,
        use_occurrence_sense_indices=False,
        source_lexical_only=False,
        include_generated_examples=True,
        include_source_context_nuance=False):
    """Return a conservative plain-text compact-response candidate.

    Invalid JSON and malformed required containers are left for the validator;
    no speculative JSON correction is attempted here.
    """
    if not isinstance(raw_text, str):
        raise TypeError("A compact provider response must be text.")
    if not isinstance(use_occurrence_sense_indices, bool):
        raise TypeError(
            "Occurrence-sense repair mode must be true or false.")
    original_sha256 = _sha256(raw_text)
    try:
        parsed = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError):
        return CompactLocalRepairResult(
            original_raw_text=raw_text,
            candidate_raw_text=raw_text,
            original_sha256=original_sha256,
            candidate_sha256=original_sha256,
            changes=())
    if not isinstance(parsed, dict):
        return CompactLocalRepairResult(
            original_raw_text=raw_text,
            candidate_raw_text=raw_text,
            original_sha256=original_sha256,
            candidate_sha256=original_sha256,
            changes=())

    changes = []
    term_results_key = process_text.SOURCE_TERM_RESULTS_KEY
    contexts_key = process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY
    rank_key = process_text.SOURCE_RANK_FIELD_NAME
    context_id_key = process_text.SOURCE_CONTEXT_ID_FIELD_NAME
    contextual_key = process_text.SOURCE_CONTEXTUAL_SENSE_KEY
    additional_key = process_text.SOURCE_ADDITIONAL_SENSES_KEY
    occurrence_indices_key = (
        process_text.SOURCE_OCCURRENCE_SENSE_INDICES_KEY)
    translation_key = process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME
    nuance_key = process_text.SOURCE_CONTEXT_NUANCE_FIELD_NAME

    _remove_extra_fields(
        parsed,
        {term_results_key, contexts_key},
        "$",
        changes)

    words_by_rank = {
        word.rank: word
        for word in chunk.words
    }
    expected_ranks = tuple(word.rank for word in chunk.words)
    expected_context_ids = tuple(
        context.context_id
        for context in chunk.contexts)
    term_results = _normalise_identity_array(
        parsed.get(term_results_key),
        identity_key=rank_key,
        expected_order=expected_ranks,
        path=f"$.{term_results_key}",
        changes=changes)
    context_translations = _normalise_identity_array(
        parsed.get(contexts_key),
        identity_key=context_id_key,
        expected_order=expected_context_ids,
        path=f"$.{contexts_key}",
        changes=changes)
    if isinstance(term_results, list):
        parsed[term_results_key] = term_results
    if isinstance(context_translations, list):
        parsed[contexts_key] = context_translations

    term_field = pipeline_store.get_language(
        pipeline.language_key).term_field
    lexical_fields = {
        pipeline_store.response_field_name(field_setting)
        for field_setting in (
            pipeline_store.get_source_lexical_field_settings(pipeline)
            if source_lexical_only
            else pipeline_store.get_requested_field_settings(pipeline))
    }
    contextual_fields = set(lexical_fields)
    additional_fields = set(lexical_fields)
    if include_generated_examples:
        additional_fields.update({
            "Sentences",
            process_text.SENTENCE_TRANSLATIONS_FIELD_NAME,
        })
    if isinstance(term_results, list):
        for result_index, result in enumerate(term_results):
            if not isinstance(result, dict):
                continue
            result_path = f"$.{term_results_key}[{result_index}]"
            allowed_result_fields = {
                rank_key,
                contextual_key,
                additional_key,
            }
            if use_occurrence_sense_indices:
                allowed_result_fields.add(occurrence_indices_key)
            _remove_extra_fields(
                result,
                allowed_result_fields,
                result_path,
                changes)
            rank = result.get(rank_key)
            word = words_by_rank.get(rank)
            if word is None:
                continue
            occurrences = tuple(
                getattr(word, "context_occurrences", ()) or ())
            if (
                    use_occurrence_sense_indices
                    and occurrence_indices_key not in result
                    and len(occurrences) <= 1):
                inferred = [0] if occurrences else []
                result[occurrence_indices_key] = inferred
                changes.append(_change(
                    "infer_unambiguous_occurrence_sense_indices",
                    (
                        f"{result_path}."
                        f"{occurrence_indices_key}"),
                    (
                        "inserted the only possible mapping for zero or one "
                        "trusted context occurrence"),
                    rank=rank,
                    occurrence_count=len(occurrences)))
            occurrence_indices = result.get(
                occurrence_indices_key)
            safe_occurrence_indices = (
                isinstance(occurrence_indices, list)
                and all(
                    not isinstance(value, bool)
                    and isinstance(value, int)
                    and value >= 0
                    for value in occurrence_indices))
            preserve_optional_cardinality = bool(
                use_occurrence_sense_indices
                and (
                    not safe_occurrence_indices
                    or any(
                        value > 0
                        for value in occurrence_indices)))
            contextual = result.get(contextual_key)
            contextual_path = f"{result_path}.{contextual_key}"
            if isinstance(contextual, dict):
                _remove_extra_fields(
                    contextual,
                    contextual_fields,
                    contextual_path,
                    changes)
                _remove_html_from_string_fields(
                    contextual,
                    contextual_fields,
                    contextual_path,
                    changes,
                    rank=rank)

            additional = result.get(additional_key)
            if not isinstance(additional, list):
                # Optional malformed additional output can be discarded
                # without touching the required contextual sense.
                if preserve_optional_cardinality:
                    # A malformed or positive occurrence map may refer to
                    # this collection. Do not erase or renumber model content
                    # when the surviving references cannot be proven.
                    continue
                result[additional_key] = []
                changes.append(_change(
                    "discard_malformed_optional_senses",
                    f"{result_path}.{additional_key}",
                    "replaced malformed optional value with an empty array",
                    rank=rank))
                continue

            retained = []
            seen_exact = []
            seen_semantic = {
                _sense_fingerprint(contextual, lexical_fields)
            }
            for sense_index, sense in enumerate(additional):
                sense_path = (
                    f"{result_path}.{additional_key}[{sense_index}]")
                if not isinstance(sense, dict):
                    if preserve_optional_cardinality:
                        retained.append(sense)
                        continue
                    changes.append(_change(
                        "discard_malformed_optional_sense",
                        sense_path,
                        "removed non-object optional sense",
                        rank=rank))
                    continue
                component = _component_only_sense(
                    sense,
                    word.surface,
                    pipeline.language_key)
                if (
                        component is not None
                        and not preserve_optional_cardinality):
                    changes.append(_change(
                        "discard_component_only_sense",
                        sense_path,
                        (
                            "removed optional sense whose examples use only "
                            "a strict component of the requested term"),
                        rank=rank,
                        component=component))
                    continue
                if _complete_term_absent_from_sense(
                        sense,
                        word.surface,
                        pipeline.language_key
                ) and not preserve_optional_cardinality:
                    changes.append(_change(
                        "discard_absent_complete_term_sense",
                        sense_path,
                        (
                            "removed an optional Classical Chinese sense "
                            "whose three examples never use the complete "
                            "requested term"),
                        rank=rank))
                    continue
                _remove_extra_fields(
                    sense,
                    additional_fields,
                    sense_path,
                    changes)
                _remove_html_from_string_fields(
                    sense,
                    lexical_fields,
                    sense_path,
                    changes,
                    rank=rank)

                for field_name in (
                        "Sentences",
                        process_text.SENTENCE_TRANSLATIONS_FIELD_NAME):
                    values = sense.get(field_name)
                    if not isinstance(values, list):
                        continue
                    for item_index, value in enumerate(values):
                        plain = _plain_text(value)
                        if plain == value:
                            continue
                        values[item_index] = plain
                        changes.append(_change(
                            "remove_model_html",
                            (
                                f"{sense_path}"
                                f"[{json.dumps(field_name, ensure_ascii=False)}]"
                                f"[{item_index}]"),
                            "preserved visible text and removed provider HTML",
                            rank=rank))

                semantic = _sense_fingerprint(
                    sense,
                    lexical_fields)
                if (
                        semantic
                        and semantic in seen_semantic
                        and not preserve_optional_cardinality):
                    changes.append(_change(
                        "discard_duplicate_optional_sense",
                        sense_path,
                        (
                            "removed an optional sense duplicating every "
                            "lexical field of an earlier sense"),
                        rank=rank))
                    continue
                if (
                        any(sense == previous for previous in seen_exact)
                        and not preserve_optional_cardinality):
                    changes.append(_change(
                        "discard_identical_optional_sense",
                        sense_path,
                        "removed exactly duplicated optional sense",
                        rank=rank))
                    continue

                retained.append(sense)
                seen_exact.append(sense)
                if semantic:
                    seen_semantic.add(semantic)
            result[additional_key] = retained

    if isinstance(context_translations, list):
        for entry_index, entry in enumerate(context_translations):
            if not isinstance(entry, dict):
                continue
            entry_path = f"$.{contexts_key}[{entry_index}]"
            _remove_extra_fields(
                entry,
                {
                    context_id_key,
                    translation_key,
                    *(
                        (nuance_key,)
                        if include_source_context_nuance
                        else ()),
                },
                entry_path,
                changes)
            for field_name in (
                    translation_key,
                    *(
                        (nuance_key,)
                        if include_source_context_nuance
                        else ())):
                value = entry.get(field_name)
                plain = _plain_text(value)
                if plain != value:
                    entry[field_name] = plain
                    changes.append(_change(
                        "remove_model_html",
                        f"{entry_path}.{field_name}",
                        "preserved visible text and removed provider HTML",
                        context_id=entry.get(context_id_key)))

    candidate_raw_text = (
        json.dumps(
            parsed,
            ensure_ascii=False,
            separators=(",", ":"))
        if changes
        else raw_text)
    return CompactLocalRepairResult(
        original_raw_text=raw_text,
        candidate_raw_text=candidate_raw_text,
        original_sha256=original_sha256,
        candidate_sha256=_sha256(candidate_raw_text),
        changes=tuple(changes))
