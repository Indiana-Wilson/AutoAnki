"""Pure selective-repair helpers for compact v9 source responses."""

from dataclasses import dataclass
import json

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
    if not request_ranks:
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
    if not words:
        raise ValueError("A compact repair request requires at least one rank.")
    return GenerationChunk(
        chunk_id=chunk.chunk_id,
        index=chunk.index,
        total=chunk.total,
        start_rank=min(word.rank for word in words),
        end_rank=max(word.rank for word in words),
        words=words,
        contexts=contexts,
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
            f"The {label} response root does not use the compact v9 shape.")
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


def merge_compact_repair(base_raw_text, repair_raw_text, chunk, scope):
    """Merge only failed v9 components and return canonical compact JSON."""
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
    all_context_ids = tuple(
        context.context_id
        for context in chunk.contexts)
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
        expected_identities=scope.request_context_ids,
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
