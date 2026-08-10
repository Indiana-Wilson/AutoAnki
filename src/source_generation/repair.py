"""Pure selective-repair helpers for compact source responses."""

from dataclasses import dataclass
import copy
import hashlib
import html
import json
import re

import pipeline_store
import process_text
from source_generation.models import GenerationChunk


@dataclass(frozen=True)
class CompactRepairScope:
    """The smallest schema-valid request and merge scope for one retry."""

    replace_ranks: tuple[int, ...]
    replace_context_ids: tuple[str, ...]
    request_ranks: tuple[int, ...]
    request_context_ids: tuple[str, ...]

    def to_dict(self):
        return {
            "replace_ranks": list(self.replace_ranks),
            "replace_context_ids": list(self.replace_context_ids),
            "request_ranks": list(self.request_ranks),
            "request_context_ids": list(self.request_context_ids),
        }


_EXAMPLE_ITEM_PATH = re.compile(
    r'^\$\.term_results\[(?P<result>\d+)\]'
    r'\.additional_senses\[(?P<sense>\d+)\]'
    r'\["(?P<field>Sentences|Sentence Translations '
    r'\(English\))"\]\[(?P<item>\d+)\]$')
_EXAMPLE_PAIR_REPAIRABLE_CODES = frozenset({
    "english_sentence_translation_contains_term",
    "missing_exact_form_source_term",
    "sentence_translation_not_english",
    "sentence_collection_item_contains_delimiter",
})
_UNSAFE_HTML = re.compile(r"<[^>]+>")
MAX_COMPACT_EXAMPLE_REPAIR_TARGETS = 64


@dataclass(frozen=True)
class CompactExampleRepairTarget:
    key: str
    source_rank: int
    additional_sense_index: int
    example_index: int
    term: str
    replace_sentence: bool
    replace_translation: bool
    sentence: str
    translation: str
    sense_fields: dict
    sibling_sentences: tuple[str, ...]
    problem_ids: tuple[str, ...]
    problem_codes: tuple[str, ...]

    def to_dict(self):
        return {
            "key": self.key,
            "source_rank": self.source_rank,
            "additional_sense_index": self.additional_sense_index,
            "example_index": self.example_index,
            "term": self.term,
            "replace_sentence": self.replace_sentence,
            "replace_translation": self.replace_translation,
            "sentence": self.sentence,
            "translation": self.translation,
            "sense_fields": dict(self.sense_fields),
            "sibling_sentences": list(self.sibling_sentences),
            "problem_ids": list(self.problem_ids),
            "problem_codes": list(self.problem_codes),
        }


@dataclass(frozen=True)
class CompactExampleRepairScope:
    base_raw_sha256: str
    targets: tuple[CompactExampleRepairTarget, ...]

    def to_dict(self):
        return {
            "base_raw_sha256": self.base_raw_sha256,
            "target_count": len(self.targets),
            "targets": [
                target.to_dict()
                for target in self.targets
            ],
        }


def derive_compact_example_repair_scope(
        report,
        chunk,
        base_raw_text,
        *,
        maximum_targets=MAX_COMPACT_EXAMPLE_REPAIR_TARGETS):
    """Derive only exact additional-sense sentence/translation pair repairs."""
    if (
            not isinstance(report, dict)
            or not isinstance(chunk, GenerationChunk)
            or not isinstance(base_raw_text, str)
            or isinstance(maximum_targets, bool)
            or not isinstance(maximum_targets, int)
            or maximum_targets < 1):
        return None
    try:
        parsed = json.loads(base_raw_text)
    except (TypeError, json.JSONDecodeError):
        return None
    term_results = (
        parsed.get(process_text.SOURCE_TERM_RESULTS_KEY)
        if isinstance(parsed, dict)
        else None)
    if not isinstance(term_results, list):
        return None
    words_by_rank = {
        word.rank: word
        for word in chunk.words
    }
    remaining = [
        problem
        for problem in report.get("problems", ())
        if (
            isinstance(problem, dict)
            and not problem.get("accepted", False))
    ]
    if not remaining:
        return None
    grouped = {}
    for problem in remaining:
        if (
                problem.get("full_retry_required")
                or problem.get("code")
                not in _EXAMPLE_PAIR_REPAIRABLE_CODES):
            return None
        path = problem.get("path")
        match = (
            _EXAMPLE_ITEM_PATH.fullmatch(path)
            if isinstance(path, str)
            else None)
        if match is None:
            return None
        result_index = int(match.group("result"))
        sense_index = int(match.group("sense"))
        item_index = int(match.group("item"))
        if not 0 <= result_index < len(term_results):
            return None
        result = term_results[result_index]
        if not isinstance(result, dict):
            return None
        rank = result.get(process_text.SOURCE_RANK_FIELD_NAME)
        word = words_by_rank.get(rank)
        if word is None:
            return None
        additional = result.get(
            process_text.SOURCE_ADDITIONAL_SENSES_KEY)
        if (
                not isinstance(additional, list)
                or not 0 <= sense_index < len(additional)
                or not isinstance(additional[sense_index], dict)):
            return None
        sense = additional[sense_index]
        sentences = sense.get("Sentences")
        translations = sense.get(
            process_text.SENTENCE_TRANSLATIONS_FIELD_NAME)
        if (
                not isinstance(sentences, list)
                or not isinstance(translations, list)
                or len(sentences) != len(translations)
                or not 0 <= item_index < len(sentences)
                or not isinstance(sentences[item_index], str)
                or not isinstance(translations[item_index], str)):
            return None
        field_name = match.group("field")
        code = problem["code"]
        if (
                code == "missing_exact_form_source_term"
                and field_name != "Sentences"):
            return None
        if (
                code in {
                    "english_sentence_translation_contains_term",
                    "sentence_translation_not_english",
                }
                and field_name
                != process_text.SENTENCE_TRANSLATIONS_FIELD_NAME):
            return None
        identity = (rank, sense_index, item_index)
        item = grouped.setdefault(
            identity,
            {
                "rank": rank,
                "word": word,
                "sense_index": sense_index,
                "item_index": item_index,
                "sense": sense,
                "sentences": sentences,
                "translations": translations,
                "replace_sentence": False,
                "replace_translation": False,
                "problem_ids": set(),
                "problem_codes": set(),
            })
        if (
                code == "missing_exact_form_source_term"
                or (
                    code == "sentence_collection_item_contains_delimiter"
                    and field_name == "Sentences")):
            item["replace_sentence"] = True
            item["replace_translation"] = True
        else:
            item["replace_translation"] = True
        if isinstance(problem.get("problem_id"), str):
            item["problem_ids"].add(problem["problem_id"])
        item["problem_codes"].add(code)

    targets = []
    for index, identity in enumerate(
            sorted(grouped)[:maximum_targets],
            start=1):
        item = grouped[identity]
        sense_fields = {
            key: value
            for key, value in item["sense"].items()
            if key not in {
                "Sentences",
                process_text.SENTENCE_TRANSLATIONS_FIELD_NAME,
            }
        }
        targets.append(CompactExampleRepairTarget(
            key=f"p{index:03d}",
            source_rank=item["rank"],
            additional_sense_index=item["sense_index"],
            example_index=item["item_index"],
            term=item["word"].surface,
            replace_sentence=item["replace_sentence"],
            replace_translation=item["replace_translation"],
            sentence=item["sentences"][item["item_index"]],
            translation=item["translations"][item["item_index"]],
            sense_fields=sense_fields,
            sibling_sentences=tuple(
                sentence
                for sibling_index, sentence in enumerate(item["sentences"])
                if sibling_index != item["item_index"]),
            problem_ids=tuple(sorted(item["problem_ids"])),
            problem_codes=tuple(sorted(item["problem_codes"])),
        ))
    if not targets:
        return None
    return CompactExampleRepairScope(
        base_raw_sha256=hashlib.sha256(
            base_raw_text.encode("utf-8")).hexdigest(),
        targets=tuple(targets),
    )


def compact_example_repair_input(scope, source_language):
    if not isinstance(scope, CompactExampleRepairScope):
        raise TypeError("A compact example repair scope is required.")
    return json.dumps(
        {
            "source_language": source_language.name,
            "exact_form_required": (
                source_language.model_language_key
                == "classical_chinese"),
            "repairs": {
                target.key: {
                    "term": target.term,
                    "sense": target.sense_fields,
                    "current_sentence": target.sentence,
                    "current_translation": target.translation,
                    "other_sentences_for_this_sense": list(
                        target.sibling_sentences),
                    "replace": [
                        field
                        for field, enabled in (
                            ("sentence", target.replace_sentence),
                            ("translation", target.replace_translation),
                        )
                        if enabled
                    ],
                    "failure_codes": list(target.problem_codes),
                }
                for target in scope.targets
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"))


def compact_example_repair_response_format(scope):
    if not isinstance(scope, CompactExampleRepairScope):
        raise TypeError("A compact example repair scope is required.")
    properties = {}
    for target in scope.targets:
        fields = {}
        required = []
        if target.replace_sentence:
            fields["sentence"] = {"type": "string"}
            required.append("sentence")
        if target.replace_translation:
            fields["translation"] = {"type": "string"}
            required.append("translation")
        properties[target.key] = {
            "type": "object",
            "properties": fields,
            "required": required,
            "additionalProperties": False,
        }
    return {
        "type": "json_schema",
        "name": "autoanki_v10_example_repair",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "repairs": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
            "required": ["repairs"],
            "additionalProperties": False,
        },
    }


def merge_compact_example_repair(
        base_raw_text,
        repair_raw_text,
        chunk,
        scope,
        pipeline):
    """Replace only approved example slots after strict local checks."""
    if not isinstance(chunk, GenerationChunk):
        raise TypeError("A source generation chunk is required.")
    if not isinstance(scope, CompactExampleRepairScope):
        raise TypeError("A compact example repair scope is required.")
    if hashlib.sha256(
            base_raw_text.encode("utf-8")).hexdigest() \
            != scope.base_raw_sha256:
        raise ValueError("The example repair base response changed.")
    try:
        base = json.loads(base_raw_text)
        repair = json.loads(repair_raw_text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Example repair inputs must be valid JSON.") from error
    if (
            not isinstance(repair, dict)
            or set(repair) != {"repairs"}
            or not isinstance(repair["repairs"], dict)
            or set(repair["repairs"])
            != {target.key for target in scope.targets}):
        raise ValueError(
            "The example repair response does not match its frozen scope.")
    source_language = pipeline_store.get_language(
        pipeline.language_key)
    term_results = base.get(
        process_text.SOURCE_TERM_RESULTS_KEY)
    if not isinstance(term_results, list):
        raise ValueError("The example repair base has no term results.")
    indexed = {}
    for result in term_results:
        if not isinstance(result, dict):
            raise ValueError("The example repair base term results are invalid.")
        rank = result.get(process_text.SOURCE_RANK_FIELD_NAME)
        if rank in indexed:
            raise ValueError("The example repair base repeats a rank.")
        indexed[rank] = result
    merged = copy.deepcopy(base)
    merged_results = {
        result[process_text.SOURCE_RANK_FIELD_NAME]: result
        for result in merged[process_text.SOURCE_TERM_RESULTS_KEY]
    }
    for target in scope.targets:
        value = repair["repairs"].get(target.key)
        expected_fields = {
            field
            for field, enabled in (
                ("sentence", target.replace_sentence),
                ("translation", target.replace_translation),
            )
            if enabled
        }
        if (
                not isinstance(value, dict)
                or set(value) != expected_fields
                or any(
                    not isinstance(item, str)
                    or not item.strip()
                    or "|" in html.unescape(item)
                    or _UNSAFE_HTML.search(item)
                    for item in value.values())):
            raise ValueError(
                f"Example repair {target.key} contains unsafe text.")
        sentence = value.get("sentence", target.sentence)
        translation = value.get(
            "translation",
            target.translation)
        if (
                source_language.model_language_key
                == "classical_chinese"
                and target.term not in html.unescape(sentence)):
            raise ValueError(
                f"Example repair {target.key} omits the complete term.")
        reason, _plain_source = (
            process_text._sentence_translation_language_issue(
                sentence,
                translation,
                source_language.model_language_key))
        if reason is not None:
            raise ValueError(
                f"Example repair {target.key} is not an English translation.")
        result = indexed.get(target.source_rank)
        merged_result = merged_results.get(target.source_rank)
        if result is None or merged_result is None:
            raise ValueError("The example repair rank is unavailable.")
        additional = result.get(
            process_text.SOURCE_ADDITIONAL_SENSES_KEY)
        merged_additional = merged_result.get(
            process_text.SOURCE_ADDITIONAL_SENSES_KEY)
        if (
                not isinstance(additional, list)
                or not isinstance(merged_additional, list)
                or not 0 <= target.additional_sense_index < len(additional)):
            raise ValueError("The example repair sense is unavailable.")
        sense = additional[target.additional_sense_index]
        merged_sense = merged_additional[
            target.additional_sense_index]
        sentences = sense.get("Sentences")
        translations = sense.get(
            process_text.SENTENCE_TRANSLATIONS_FIELD_NAME)
        merged_sentences = merged_sense.get("Sentences")
        merged_translations = merged_sense.get(
            process_text.SENTENCE_TRANSLATIONS_FIELD_NAME)
        item_index = target.example_index
        if (
                not all(
                    isinstance(values, list)
                    for values in (
                        sentences,
                        translations,
                        merged_sentences,
                        merged_translations))
                or not 0 <= item_index < len(sentences)
                or len(sentences) != len(translations)
                or sentences[item_index] != target.sentence
                or translations[item_index] != target.translation):
            raise ValueError("The example repair target became stale.")
        if target.replace_sentence:
            merged_sentences[item_index] = sentence
        if target.replace_translation:
            merged_translations[item_index] = translation
    return json.dumps(
        merged,
        ensure_ascii=False,
        separators=(",", ":"))


def derive_compact_repair_scope(report, chunk):
    """Return a safe selective scope, or ``None`` when a full retry is needed."""
    if not isinstance(report, dict) or not isinstance(
            chunk,
            GenerationChunk):
        return None
    problems = report.get("problems")
    if not isinstance(problems, list):
        return None
    remaining = [
        problem
        for problem in problems
        if (
            isinstance(problem, dict)
            and not problem.get("accepted", False))
    ]
    if not remaining:
        return None

    words_by_rank = {
        word.rank: word
        for word in chunk.words
    }
    contexts_by_id = {
        context.context_id: context
        for context in chunk.contexts
    }
    replace_ranks = set()
    replace_context_ids = set()
    for problem in remaining:
        if problem.get("full_retry_required"):
            return None
        target = problem.get("repair_target")
        if not isinstance(target, dict):
            return None
        kind = target.get("kind")
        if kind == "source_rank":
            rank = target.get("source_rank")
            if (
                    isinstance(rank, bool)
                    or not isinstance(rank, int)
                    or rank not in words_by_rank):
                return None
            replace_ranks.add(rank)
        elif kind == "context":
            context_id = target.get("context_id")
            if (
                    not isinstance(context_id, str)
                    or context_id not in contexts_by_id):
                return None
            replace_context_ids.add(context_id)
        else:
            return None

    request_ranks = set(replace_ranks)
    request_ranks.update(
        word.rank
        for word in chunk.words
        if word.context_id in replace_context_ids)
    if not request_ranks and not replace_context_ids:
        return None
    request_context_ids = set(replace_context_ids)
    request_context_ids.update(
        words_by_rank[rank].context_id
        for rank in request_ranks
        if words_by_rank[rank].context_id is not None)
    if not request_context_ids <= set(contexts_by_id):
        return None

    rank_order = {
        word.rank: index
        for index, word in enumerate(chunk.words)
    }
    context_order = {
        context.context_id: index
        for index, context in enumerate(chunk.contexts)
    }
    return CompactRepairScope(
        replace_ranks=tuple(sorted(
            replace_ranks,
            key=rank_order.__getitem__)),
        replace_context_ids=tuple(sorted(
            replace_context_ids,
            key=context_order.__getitem__)),
        request_ranks=tuple(sorted(
            request_ranks,
            key=rank_order.__getitem__)),
        request_context_ids=tuple(sorted(
            request_context_ids,
            key=context_order.__getitem__)),
    )


def build_compact_repair_chunk(chunk, scope):
    """Build one request subset while preserving immutable source identities."""
    if not isinstance(chunk, GenerationChunk):
        raise TypeError("A source generation chunk is required.")
    if not isinstance(scope, CompactRepairScope):
        raise TypeError("A compact repair scope is required.")
    ranks = set(scope.request_ranks)
    context_ids = set(scope.request_context_ids)
    words = tuple(
        word
        for word in chunk.words
        if word.rank in ranks)
    contexts = tuple(
        context
        for context in chunk.contexts
        if context.context_id in context_ids)
    if {word.rank for word in words} != ranks:
        raise ValueError("Repair ranks do not match the saved source chunk.")
    if {context.context_id for context in contexts} != context_ids:
        raise ValueError("Repair contexts do not match the saved source chunk.")
    return GenerationChunk(
        chunk_id=chunk.chunk_id,
        index=chunk.index,
        total=chunk.total,
        start_rank=(
            min(word.rank for word in words)
            if words
            else chunk.start_rank),
        end_rank=(
            max(word.rank for word in words)
            if words
            else chunk.end_rank),
        words=words,
        contexts=contexts,
        source_context_translation_ids=(
            tuple(scope.replace_context_ids)
            if chunk.source_context_translation_ids is not None
            else None),
    )


def _indexed_items(
        parsed,
        *,
        root_key,
        identity_key,
        expected_identities,
        label,
        replace_identities=()):
    if not isinstance(parsed, dict):
        raise ValueError(f"The {label} response root is not an object.")
    expected_root = {
        process_text.SOURCE_TERM_RESULTS_KEY,
        process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY,
    }
    if set(parsed) != expected_root:
        raise ValueError(
            f"The {label} response root does not use the compact source shape.")
    items = parsed.get(root_key)
    if not isinstance(items, list):
        raise ValueError(f"The {label} {root_key} value is not an array.")
    expected = set(expected_identities)
    replaceable = set(replace_identities)
    if not replaceable <= expected:
        raise ValueError("Repair identities are outside the saved scope.")
    indexed = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"The {label} {root_key} array is malformed.")
        identity = item.get(identity_key)
        try:
            known_identity = identity in expected
        except TypeError as error:
            raise ValueError(
                f"The {label} {root_key} identity is malformed.") from error
        if not known_identity:
            raise ValueError(
                f"The {label} {root_key} contains an unknown identity.")
        if identity in indexed:
            if identity in replaceable:
                continue
            raise ValueError(
                f"The {label} {root_key} array repeats {identity!r}.")
        indexed[identity] = item
    if set(indexed) - replaceable != expected - replaceable:
        raise ValueError(
            f"The {label} {root_key} identities do not match their scope.")
    return indexed


def merge_compact_repair(
        base_raw_text,
        repair_raw_text,
        chunk,
        scope,
        *,
        remembered_context_ids=()):
    """Merge only failed compact components and return canonical JSON."""
    if not isinstance(chunk, GenerationChunk):
        raise TypeError("A source generation chunk is required.")
    if not isinstance(scope, CompactRepairScope):
        raise TypeError("A compact repair scope is required.")
    try:
        base = json.loads(base_raw_text)
        repair = json.loads(repair_raw_text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Compact repair inputs must be valid JSON.") from error

    rank_key = process_text.SOURCE_RANK_FIELD_NAME
    context_id_key = process_text.SOURCE_CONTEXT_ID_FIELD_NAME
    term_key = process_text.SOURCE_TERM_RESULTS_KEY
    context_key = process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY
    all_ranks = tuple(word.rank for word in chunk.words)
    requested_context_ids = tuple(
        chunk.requested_source_context_ids)
    remembered_context_ids = set(remembered_context_ids)
    if not remembered_context_ids <= set(requested_context_ids):
        raise ValueError(
            "Remembered repair contexts are outside the saved chunk.")
    if remembered_context_ids & set(scope.replace_context_ids):
        raise ValueError(
            "A remembered source translation cannot be provider-repaired.")
    all_context_ids = tuple(
        context_id
        for context_id in requested_context_ids
        if context_id not in remembered_context_ids)
    base_terms = _indexed_items(
        base,
        root_key=term_key,
        identity_key=rank_key,
        expected_identities=all_ranks,
        label="base",
        replace_identities=scope.replace_ranks)
    base_contexts = _indexed_items(
        base,
        root_key=context_key,
        identity_key=context_id_key,
        expected_identities=all_context_ids,
        label="base",
        replace_identities=scope.replace_context_ids)
    repair_terms = _indexed_items(
        repair,
        root_key=term_key,
        identity_key=rank_key,
        expected_identities=scope.request_ranks,
        label="repair")
    repair_contexts = _indexed_items(
        repair,
        root_key=context_key,
        identity_key=context_id_key,
        expected_identities=(
            scope.replace_context_ids
            if chunk.source_context_translation_ids is not None
            else scope.request_context_ids),
        label="repair")

    replace_ranks = set(scope.replace_ranks)
    replace_context_ids = set(scope.replace_context_ids)
    merged = {
        term_key: [
            (
                repair_terms[rank]
                if rank in replace_ranks
                else base_terms[rank])
            for rank in all_ranks
        ],
        context_key: [
            (
                repair_contexts[context_id]
                if context_id in replace_context_ids
                else base_contexts[context_id])
            for context_id in all_context_ids
        ],
    }
    return json.dumps(
        merged,
        ensure_ascii=False,
        separators=(",", ":"))
