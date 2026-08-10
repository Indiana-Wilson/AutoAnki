import json
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk

import anki_integration
import credential_store
import gui_preferences
import learned_filter_store
import pipeline_runner
import pipeline_store
import prompt_builder
import process_text
from source_generation.model_catalog import (
    is_local_source_model,
    source_model_profile,
)


SOURCE_CONTEXT_OPTIONS = (
    (
        "none",
        "No source context",
        "Send only the ordered vocabulary for each request.",
    ),
    (
        "sentence",
        "Current sentence",
        (
            "Share each distinct sentence once, then point every overlapping "
            "word to it."
        ),
    ),
    (
        "sentence_neighbors",
        "Current ± previous and next sentence",
        (
            "Merge overlapping three-sentence windows and send each resulting "
            "context only once."
        ),
    ),
    (
        "chunk_span",
        "Complete source span for the chunk",
        (
            "Send one continuous span from the earliest first occurrence to "
            "the latest first occurrence in each chunk."
        ),
    ),
)
SOURCE_CONTEXT_LABELS = {
    key: label
    for key, label, _description in SOURCE_CONTEXT_OPTIONS
}
SOURCE_CONTEXT_KEYS_BY_LABEL = {
    label: key
    for key, label, _description in SOURCE_CONTEXT_OPTIONS
}
SOURCE_CONTEXT_DESCRIPTIONS = {
    key: description
    for key, _label, description in SOURCE_CONTEXT_OPTIONS
}
SOURCE_PROTOCOL_OPTIONS = (
    ("v10", "Compact v10 · local-first"),
    ("v9", "Compact v9 rollback"),
    ("v8", "Legacy v8 rollback"),
)
SOURCE_PROTOCOL_KEYS_BY_LABEL = {
    label: key
    for key, label in SOURCE_PROTOCOL_OPTIONS
}
SOURCE_PROTOCOL_ESTIMATE_LABELS = {
    "v10": "compact v10 local-first",
    "v9": "compact v9 rollback",
    "v8": "legacy v8 rollback",
}
SOURCE_REASONING_OPTIONS = (
    ("low", "Low · recommended"),
    ("none", "None · lowest cost, small batches"),
)
SOURCE_REASONING_KEYS_BY_LABEL = {
    label: key
    for key, label in SOURCE_REASONING_OPTIONS
}
SOURCE_EXECUTION_OPTIONS = (
    ("standard", "Standard · immediate"),
    ("economy", "Economy Batch · 50% token price"),
)
SOURCE_EXECUTION_KEYS_BY_LABEL = {
    label: key
    for key, label in SOURCE_EXECUTION_OPTIONS
}
SOURCE_MODEL_OPTIONS = (
    ("gpt-5.4-mini", "GPT-5.4 mini · proven default"),
)
SOURCE_MODEL_KEYS_BY_LABEL = {
    label: key
    for key, label in SOURCE_MODEL_OPTIONS
}

ENHANCED_AUDIO_BACKEND_LABELS = {
    "japanese": "MeloTTS · JP",
    "english": "Fun-CosyVoice3 0.5B",
    "classical_chinese": "Kokoro 82M · zm_010",
    "french": "Fun-CosyVoice3 0.5B",
    "latin": "No supported local voice",
}


def enhanced_audio_backend_label(language_key):
    """Return the exact production model and voice shown in card setup."""
    return ENHANCED_AUDIO_BACKEND_LABELS[language_key]

SOURCE_JOB_HEADINGS = {
    "job": "Job",
    "source": "Source",
    "chunk": "Chunk / stage",
    "worker": "Worker",
    "status": "Status",
    "attempts": "Attempts",
    "detail": "Latest detail",
}
SOURCE_JOB_COLUMNS = (
    "source",
    "chunk",
    "worker",
    "status",
    "attempts",
    "detail",
)
SOURCE_JOB_DETAIL_MIN_WIDTH = 260
SOURCE_JOBS_TREE_VISIBLE_ROWS = 30
SOURCE_JOB_INSPECTION_VISIBLE_LINES = 30
SOURCE_VALIDATION_DIALOG_PREFERRED_WIDTH = 1900
SOURCE_VALIDATION_DIALOG_PREFERRED_HEIGHT = 1320
SOURCE_VALIDATION_DIALOG_MIN_WIDTH = 760
SOURCE_VALIDATION_DIALOG_MIN_HEIGHT = 620
SOURCE_VALIDATION_DIALOG_MARGIN = 60

SOURCE_JOB_COLUMN_MIN_WIDTHS = {
    "job": 120,
    "source": 130,
    "chunk": 90,
    "worker": 110,
    "status": 105,
    "attempts": 80,
    "detail": 150,
}
SOURCE_JOB_COLUMN_MAX_WIDTHS = {
    "job": 300,
    "source": 320,
    "chunk": 170,
    "worker": 230,
    "status": 220,
    "attempts": 115,
}


def _source_job_value(job, name, default=""):
    if isinstance(job, dict):
        return job.get(name, default)
    return getattr(job, name, default)


def source_job_parent_id(job, fallback=""):
    """Return the deck-generation job owning one displayed stage."""
    parent_id = _source_job_value(job, "parent_job_id", "")
    if parent_id not in (None, ""):
        return str(parent_id)
    row_id = _source_job_value(
        job,
        "job_id",
        _source_job_value(job, "id", fallback))
    row_id = str(row_id)
    if "::" in row_id:
        return row_id.split("::", 1)[0]
    return row_id


def group_source_job_rows(jobs):
    """Group saved request/finalization rows without losing job order.

    The workflow adapter historically returned every request row followed by
    every finalization row. Grouping at the display boundary keeps each Deck
    row with the requests it finalizes, including for already-saved jobs.
    """
    grouped = {}
    order = []
    for index, job in enumerate(jobs):
        parent_id = source_job_parent_id(job, fallback=index)
        if parent_id not in grouped:
            grouped[parent_id] = []
            order.append(parent_id)
        grouped[parent_id].append(job)

    result = []
    for parent_id in order:
        rows = grouped[parent_id]
        regular = []
        finalization = []
        for row in rows:
            row_id = str(_source_job_value(row, "job_id", ""))
            chunk = str(_source_job_value(
                row,
                "chunk_label",
                _source_job_value(row, "chunk", "")))
            target = (
                finalization
                if (
                    bool(_source_job_value(
                        row,
                        "is_finalization_stage",
                        False))
                    or row_id.endswith("::finalize")
                    or chunk == "Deck")
                else regular)
            target.append(row)
        result.append((parent_id, tuple((*regular, *finalization))))
    return tuple(result)


def source_job_group_label(parent_id, ordinal):
    """Create a short but recognizable label for a generation job."""
    parent_id = str(parent_id)
    stamp, separator, suffix = parent_id.partition("-")
    if (
            separator
            and len(stamp) == 15
            and stamp[8] == "T"
            and stamp[:8].isdigit()
            and stamp[9:].isdigit()):
        timestamp = (
            f"{stamp[4:6]}-{stamp[6:8]} "
            f"{stamp[9:11]}:{stamp[11:13]}")
        return f"{timestamp} · {suffix[:8]}"
    if len(parent_id) > 22:
        parent_id = f"{parent_id[:19]}…"
    return f"{ordinal} · {parent_id}"


def source_job_column_widths(rows, measure, available_width):
    """Fit every job column into the viewport without a horizontal sprawl."""
    fixed_columns = ("job", *SOURCE_JOB_COLUMNS[:-1])
    widths = {}
    for column in fixed_columns:
        candidates = (
            SOURCE_JOB_HEADINGS[column],
            *(
                str(row.get(column, ""))
                for row in rows),
        )
        measured = max(measure(value) for value in candidates) + 24
        widths[column] = min(
            SOURCE_JOB_COLUMN_MAX_WIDTHS[column],
            max(SOURCE_JOB_COLUMN_MIN_WIDTHS[column], measured))

    target = max(1, int(available_width) - 4)
    widths["detail"] = max(
        SOURCE_JOB_COLUMN_MIN_WIDTHS["detail"],
        SOURCE_JOB_DETAIL_MIN_WIDTH)
    overflow = sum(widths.values()) - target
    if overflow > 0:
        # Shrink the descriptive columns first, retaining enough space for
        # every heading. The selected row's unabridged detail remains in the
        # reader directly below the table.
        shrink_order = (
            "source",
            "job",
            "worker",
            "status",
            "chunk",
            "detail",
            "attempts",
        )
        for column in shrink_order:
            capacity = (
                widths[column]
                - SOURCE_JOB_COLUMN_MIN_WIDTHS[column])
            reduction = min(overflow, max(0, capacity))
            widths[column] -= reduction
            overflow -= reduction
            if overflow <= 0:
                break
    if overflow > 0:
        # Extremely narrow displays are still usable: distribute the last
        # reduction instead of allowing the Treeview to force the entire page
        # wider than its notebook viewport.
        columns = (*fixed_columns, "detail")
        while overflow > 0:
            changed = False
            for column in columns:
                if widths[column] > 40:
                    widths[column] -= 1
                    overflow -= 1
                    changed = True
                    if overflow <= 0:
                        break
            if not changed:
                break
    else:
        widths["detail"] += -overflow
    return widths


def validation_problem_column_widths(measure, available_width):
    """Return readable problem-list columns that exactly fit its viewport."""
    available_width = max(1, int(available_width) - 4)
    decision_preferred = max(
        measure("Review state"),
        measure("Can accept"),
        measure("Accepted"),
        measure("Locked")) + 28
    decision = min(
        decision_preferred,
        max(1, int(available_width * 0.23)))
    remaining = max(0, available_width - decision)
    location = min(
        max(1, int(available_width * 0.34)),
        remaining)
    problem = max(0, remaining - location)
    return {
        "decision": decision,
        "location": location,
        "problem": problem,
    }


def source_validation_dialog_geometry(screen_width, screen_height):
    """Fit the problem reviewer inside the current display.

    Tk can report fairly small work areas for remote desktops and scaled
    displays.  A preferred or minimum size must therefore never be allowed to
    exceed the display itself.
    """
    screen_width = max(1, int(screen_width))
    screen_height = max(1, int(screen_height))
    horizontal_margin = min(
        SOURCE_VALIDATION_DIALOG_MARGIN,
        max(0, (screen_width - 1) // 2))
    vertical_margin = min(
        SOURCE_VALIDATION_DIALOG_MARGIN,
        max(0, (screen_height - 1) // 2))
    width = min(
        SOURCE_VALIDATION_DIALOG_PREFERRED_WIDTH,
        max(1, screen_width - (2 * horizontal_margin)))
    height = min(
        SOURCE_VALIDATION_DIALOG_PREFERRED_HEIGHT,
        max(1, screen_height - (2 * vertical_margin)))
    return (
        width,
        height,
        max((screen_width - width) // 2, 0),
        max((screen_height - height) // 2, 0),
    )


def _source_inspection_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=False,
        indent=2,
        default=str)


def _source_inspection_label(name):
    labels = {
        "request_view": "REQUEST OVERVIEW AND ACTUAL BATCH INPUT",
        "request": "SAVED INPUT FOR THIS REQUEST CHUNK",
        "request_contract": "OPENAI REQUEST SETTINGS AND RESPONSE SCHEMA",
        "request_contract_origin": "WHERE THOSE REQUEST SETTINGS CAME FROM",
        "response_view": "LATEST RESPONSE AND VALIDATION RESULT",
        "status": "CURRENT CHUNK STATUS",
        "attempts": "SAVED RESPONSE ATTEMPTS (OLDEST FIRST)",
        "validation": "HUMAN-REVIEWABLE VALIDATION RESULT",
        "workflow": "PACKAGING / IMPORT RESULT",
        "combined": "VALIDATED CARD BATCH USED FOR FINALIZATION",
        "job_path": "SAVED JOB FOLDER",
    }
    return labels.get(name, name.replace("_", " ").upper())


def format_source_inspection_fields(fields, empty_message):
    """Format top-level inspection fields as clearly labelled sections."""
    if not fields:
        return empty_message
    sections = []
    for name, value in fields:
        label = _source_inspection_label(name)
        if isinstance(value, str):
            body = value
        else:
            body = _source_inspection_json(value)
        sections.append(f"{label}\n{'=' * len(label)}\n{body}")
    return "\n\n".join(sections)


def _problem_value_text(value):
    if value is None:
        return "Not specified"
    if isinstance(value, str):
        return value
    return _source_inspection_json(value)


def validation_report_summary(report):
    """Explain whether and how one retained response can be resolved."""
    if not isinstance(report, dict) or not report.get("available", True):
        return (
            "No retained model response is available for validation. "
            "There is nothing that can be manually accepted."
        )
    problem_count = int(report.get("problem_count", 0))
    accepted_count = int(report.get("accepted_problem_count", 0))
    remaining_count = int(
        report.get(
            "remaining_problem_count",
            max(0, problem_count - accepted_count)))
    if not problem_count:
        return (
            "No validation problems were found. The retained response is "
            "valid."
        )
    counts = (
        f"{problem_count:,} problem"
        f"{'' if problem_count == 1 else 's'} found"
        f" · {accepted_count:,} manually accepted"
        f" · {remaining_count:,} remaining."
    )
    if remaining_count == 0:
        return (
            counts
            + " Every recorded issue has been reviewed and accepted; JSON "
            "syntax and card structure still passed."
        )
    if not report.get("syntax_valid"):
        return (
            counts
            + " The response is not valid JSON, so it cannot be manually "
            "accepted. Retry it or repair the JSON."
        )
    if not report.get("structurally_valid"):
        return (
            counts
            + " The JSON parses, but its card structure is unsafe. Structural "
            "problems cannot be overridden; retry or repair the response."
        )
    if report.get("can_complete_with_manual_acceptance"):
        return (
            counts
            + " JSON syntax and card structure are valid. Every remaining "
            "item is a content-rule issue that may be accepted after review."
        )
    return (
        counts
        + " At least one remaining structural problem is locked and cannot "
        "be overridden."
    )


def format_validation_problem_details(problem):
    """Render one validation problem for a human reviewer."""
    if not isinstance(problem, dict):
        return str(problem)
    if problem.get("accepted"):
        decision = "Accepted manually"
    elif problem.get("overrideable"):
        decision = (
            "Reviewable content issue — may be accepted if the card is "
            "genuinely usable"
        )
    else:
        decision = (
            "Locked structural issue — cannot be manually accepted"
        )
    sections = (
        ("STATUS", decision),
        ("WHERE", problem.get("location") or "Response"),
        ("WHAT IS WRONG", problem.get("title") or "Validation problem"),
        ("WHY", problem.get("message") or "No explanation was provided."),
        ("EXPECTED", _problem_value_text(problem.get("expected"))),
        ("ACTUAL", _problem_value_text(problem.get("actual"))),
        (
            "SUGGESTED NEXT STEP",
            problem.get("suggestion")
            or "Retry the request or review the affected response section."),
        ("EXACT JSON PATH", problem.get("path") or "$"),
    )
    return "\n\n".join(
        f"{heading}\n{'=' * len(heading)}\n{body}"
        for heading, body in sections)


def latest_source_response_text(inspection):
    """Return the retained raw text for the latest inspected attempt."""
    if not isinstance(inspection, dict):
        return None
    response_view = inspection.get("response_view")
    if isinstance(response_view, dict):
        latest = response_view.get("latest_attempt")
        if isinstance(latest, dict) and isinstance(
                latest.get("raw_text"), str):
            return latest["raw_text"]
    status = inspection.get("status")
    latest_number = None
    if isinstance(status, dict):
        latest_path = status.get("latest_attempt_path")
        if isinstance(latest_path, str) and latest_path:
            try:
                latest_number = int(Path(latest_path).name)
            except ValueError:
                latest_number = None
    attempts = inspection.get("attempts", ())
    if isinstance(attempts, (tuple, list)):
        ordered = tuple(
            attempt
            for attempt in attempts
            if isinstance(attempt, dict)
            and isinstance(attempt.get("raw_text"), str))
        if latest_number is not None:
            for attempt in ordered:
                if attempt.get("attempt") == latest_number:
                    return attempt["raw_text"]
        if ordered:
            return ordered[-1]["raw_text"]
    return None


def inspection_validation_report(inspection):
    if not isinstance(inspection, dict):
        return None
    report = inspection.get("validation")
    if isinstance(report, dict):
        return report
    response_view = inspection.get("response_view")
    if isinstance(response_view, dict):
        report = response_view.get("validation")
        if isinstance(report, dict):
            return report
    return None


def validation_problem_affected_section(inspection, problem):
    """Show the exact affected card when the response is parseable."""
    raw_text = latest_source_response_text(inspection)
    if raw_text is None:
        return "No retained response text is available."
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return raw_text
    if (
            isinstance(parsed, dict)
            and (
                "contextual_cards" in parsed
                or "additional_sense_cards" in parsed)):
        contextual_cards = parsed.get("contextual_cards")
        additional_cards = parsed.get("additional_sense_cards")
        cards = [
            *(
                contextual_cards
                if isinstance(contextual_cards, list)
                else ()),
            *(
                additional_cards
                if isinstance(additional_cards, list)
                else ()),
        ]
    elif (
            isinstance(parsed, dict)
            and isinstance(parsed.get("term_results"), dict)):
        cards = []
        term_results = parsed["term_results"]

        def rank_order(item):
            rank_text = item[0]
            try:
                return (0, int(rank_text))
            except (TypeError, ValueError):
                return (1, str(rank_text))

        for rank, group in sorted(
                term_results.items(),
                key=rank_order):
            if not isinstance(group, dict):
                continue
            contextual = group.get("contextual_sense")
            if isinstance(contextual, dict):
                cards.append({
                    "rank": rank,
                    "role": "contextual_sense",
                    "sense": contextual,
                })
            additional = group.get("additional_senses")
            if isinstance(additional, list):
                cards.extend(
                    {
                        "rank": rank,
                        "role": "additional_sense",
                        "additional_sense_number": index + 1,
                        "sense": sense,
                    }
                    for index, sense in enumerate(additional)
                    if isinstance(sense, dict))
    elif (
            isinstance(parsed, dict)
            and isinstance(parsed.get("term_results"), list)):
        cards = []
        for group in parsed["term_results"]:
            if not isinstance(group, dict):
                continue
            rank = group.get("rank")
            contextual = group.get("contextual_sense")
            if isinstance(contextual, dict):
                cards.append({
                    "rank": rank,
                    "role": "contextual_sense",
                    "sense": contextual,
                })
            additional = group.get("additional_senses")
            if isinstance(additional, list):
                cards.extend(
                    {
                        "rank": rank,
                        "role": "additional_sense",
                        "additional_sense_number": index + 1,
                        "sense": sense,
                    }
                    for index, sense in enumerate(additional)
                    if isinstance(sense, dict))
    else:
        cards = (
            parsed.get("cards")
            if isinstance(parsed, dict)
            else parsed)
    card_index = (
        problem.get("card_index")
        if isinstance(problem, dict)
        else None)
    if (
            isinstance(card_index, int)
            and isinstance(cards, list)
            and 0 <= card_index < len(cards)):
        return _source_inspection_json(cards[card_index])
    if (
            isinstance(problem, dict)
            and problem.get("scope") == "request"):
        return (
            "This problem concerns a requested source term rather than a "
            "returned card.\n\n"
            + _problem_value_text({
                "term": problem.get("term"),
                "expected": problem.get("expected"),
                "actual": problem.get("actual"),
            })
        )
    return _source_inspection_json(parsed)


def source_job_inspection_texts(value):
    """Split one inspected stage into explicit request and response text."""
    if not isinstance(value, dict):
        return (
            "No separately saved request is available for this item.",
            str(value),
        )

    is_finalization = "workflow" in value and "request" not in value
    if is_finalization:
        request_names = (
            "request_contract",
            "request_contract_origin",
            "combined",
            "job_path",
        )
        response_names = ("workflow",)
    elif "request_view" in value or "response_view" in value:
        request_names = (
            "request_view",
            "request_contract_origin",
        )
        response_names = (
            "response_view",
            "attempts",
        )
    else:
        request_names = (
            "request_view",
            "request",
            "request_contract",
            "request_contract_origin",
        )
        response_names = (
            "response_view",
            "validation",
            "status",
            "attempts",
            "response",
            "result",
            "output",
            "validation_problems",
            "problems",
            "error",
        )

    request_fields = [
        (name, value[name])
        for name in request_names
        if name in value]
    response_fields = [
        (name, value[name])
        for name in response_names
        if name in value]
    assigned = {
        name
        for name, _field_value in (*request_fields, *response_fields)}
    assigned.add("scope")
    if "request_view" in value or "response_view" in value:
        assigned.update({
            "request",
            "request_contract",
            "status",
            "validation",
        })
    for name, field_value in value.items():
        if name not in assigned:
            response_fields.append((name, field_value))

    return (
        format_source_inspection_fields(
            request_fields,
            (
                "No separately saved request is available for this stage. "
                "Use the Response tab for its saved result."
            )),
        format_source_inspection_fields(
            response_fields,
            "No response attempt or result has been saved yet."),
    )


def source_job_inspection_scope(record):
    """Explain exactly how much of a generation run is being inspected."""
    parent_id = str(record.get("parent_job_id", record.get("job_id", "")))
    source = str(record.get("source", "")).strip()
    chunk = str(record.get("chunk", "")).strip()
    if chunk == "Deck" or str(record.get("job_id", "")).endswith(
            "::finalize"):
        subject = (
            "one Deck packaging/import stage from one deck-generation job; "
            "this is not an OpenAI request")
    else:
        subject = (
            "one saved OpenAI request chunk from one deck-generation job; "
            "this is not the whole job")
    qualifiers = [
        value
        for value in (
            source,
            f"chunk {chunk}" if chunk and chunk != "Deck" else "",
            parent_id,
        )
        if value]
    suffix = f" ({' · '.join(qualifiers)})" if qualifiers else ""
    return f"Showing {subject}{suffix}."


@dataclass(frozen=True)
class SourceUiOption:
    """One prepared source presented by the Generate UI."""

    key: str
    name: str
    word_count: int | None = None
    token_occurrence_count: int | None = None
    section_count: int | None = None
    source_language_key: str = ""
    preset: bool = False


@dataclass(frozen=True)
class SourcePreviewItem:
    """One ordered vocabulary candidate and its retained local context."""

    rank: int
    term: str
    section_title: str
    previous_sentence: str
    current_sentence: str
    next_sentence: str


@dataclass(frozen=True)
class SourcePreviewPage:
    """A validated page returned by the source-preview callback."""

    source_key: str
    total: int
    offset: int
    items: tuple[SourcePreviewItem, ...]


@dataclass(frozen=True)
class ManualInputFilterResult:
    """Validated result of an exact learned-word lookup."""

    filtered_text: str
    excluded_count: int
    remaining_count: int
    added_character_count: int = 0


BUILT_IN_SOURCE_OPTIONS = (
    SourceUiOption(
        key="daodejing_wang_bi",
        name="Daodejing [Wang Bi]",
        word_count=944,
        section_count=81,
        source_language_key="classical_chinese_wang_bi",
        preset=True),
    SourceUiOption(
        key="daodejing_mawangdui",
        name="Daodejing [Mawangdui]",
        word_count=991,
        section_count=81,
        source_language_key="classical_chinese_han",
        preset=True),
    SourceUiOption(
        key="journey_to_the_west",
        name="Journey to the West",
        word_count=24224,
        section_count=100,
        source_language_key="classical_chinese_ming",
        preset=True),
)

PREPARABLE_SOURCE_LANGUAGE_NAMES = (
    "Classical Chinese (Early Han)",
    "Classical Chinese (Wang Bi recension)",
    "Classical Chinese (Warring States)",
    "Classical Chinese (Ming)",
    "Middle English",
    "Old English",
)


def normalise_source_preview_response(value):
    """Validate the small mapping contract used by the source inspector."""
    if not isinstance(value, dict):
        raise TypeError("Source preview results must be returned as an object.")

    source_key = value.get("source_key")
    if not isinstance(source_key, str) or not source_key.strip():
        raise ValueError("Source preview results require a source_key.")
    source_key = source_key.strip()

    total = value.get("total")
    offset = value.get("offset")
    for name, number in (("total", total), ("offset", offset)):
        if (
                isinstance(number, bool)
                or not isinstance(number, int)
                or number < 0):
            raise ValueError(
                f"Source preview {name} must be a non-negative integer.")
    if offset > total:
        raise ValueError(
            "Source preview offset cannot be greater than its total.")

    raw_items = value.get("items")
    if not isinstance(raw_items, (tuple, list)):
        raise ValueError("Source preview items must be a list.")
    if offset + len(raw_items) > total:
        raise ValueError(
            "Source preview items extend beyond the reported total.")
    if offset < total and not raw_items:
        raise ValueError(
            "Source preview returned an empty page before the end.")

    items = []
    last_rank = 0
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("Every source preview item must be an object.")
        rank = raw_item.get("rank")
        if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank < 1
                or rank <= last_rank):
            raise ValueError(
                "Source preview ranks must be positive and strictly ordered.")
        term = raw_item.get("term")
        if not isinstance(term, str) or not term.strip():
            raise ValueError(
                "Every source preview item requires a non-empty term.")
        text_fields = {}
        for field_name in (
                "section_title",
                "previous_sentence",
                "current_sentence",
                "next_sentence"):
            field_value = raw_item.get(field_name, "")
            if field_value is None:
                field_value = ""
            if not isinstance(field_value, str):
                raise ValueError(
                    "Source preview sentence and section values must be text.")
            text_fields[field_name] = field_value.strip()
        items.append(SourcePreviewItem(
            rank=rank,
            term=term.strip(),
            **text_fields))
        last_rank = rank

    return SourcePreviewPage(
        source_key=source_key,
        total=total,
        offset=offset,
        items=tuple(items))


def normalise_manual_input_filter_response(value, candidate_count):
    """Validate the learned-word filter without touching OpenAI."""
    if not isinstance(value, dict):
        raise TypeError("Manual-input filter results must be an object.")
    if (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count < 0):
        raise ValueError("Candidate count must be a non-negative integer.")
    filtered_text = value.get("filtered_text")
    if not isinstance(filtered_text, str):
        raise ValueError(
            "Manual-input filter results require filtered_text.")
    excluded_count = value.get("excluded_count")
    remaining_count = value.get("remaining_count")
    added_character_count = value.get("added_character_count", 0)
    for name, number in (
            ("excluded_count", excluded_count),
            ("remaining_count", remaining_count),
            ("added_character_count", added_character_count)):
        if (
                isinstance(number, bool)
                or not isinstance(number, int)
                or number < 0):
            raise ValueError(
                f"Manual-input filter {name} must be a non-negative integer.")
    filtered_candidates = tuple(
        line.strip()
        for line in filtered_text.splitlines()
        if line.strip())
    if len(filtered_candidates) != remaining_count:
        raise ValueError(
            "Manual-input filter remaining_count does not match "
            "filtered_text.")
    if (
            excluded_count + remaining_count
            != candidate_count + added_character_count):
        raise ValueError(
            "Manual-input filter counts do not match the requested lines.")
    return ManualInputFilterResult(
        filtered_text="\n".join(filtered_candidates),
        excluded_count=excluded_count,
        remaining_count=remaining_count,
        added_character_count=added_character_count)


def parse_source_chunk_size(value, *, maximum=10000):
    """Validate the request size without accepting booleans or decimals."""
    text = str(value).strip()
    if not text or not text.isdecimal():
        raise ValueError("Chunk size must be a positive whole number.")
    chunk_size = int(text)
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least 1.")
    if chunk_size > maximum:
        raise ValueError(
            f"Chunk size cannot exceed {maximum:,} words per request.")
    return chunk_size


def parse_source_prefix_token_limit(value, *, maximum=10_000_000):
    """Validate a running-text prefix measured in token occurrences."""
    text = str(value).strip()
    if not text or not text.isdecimal():
        raise ValueError(
            "Source prefix length must be a positive whole number.")
    token_limit = int(text)
    if token_limit < 1:
        raise ValueError("Source prefix length must be at least 1 word.")
    if token_limit > maximum:
        raise ValueError(
            "Source prefix length cannot exceed "
            f"{maximum:,} token occurrences.")
    return token_limit


def parse_request_stagger_ms(value, *, maximum=60000):
    """Validate an optional delay between starting paid requests."""
    text = str(value).strip()
    if not text or not text.isdecimal():
        raise ValueError(
            "Request stagger must be a non-negative whole number.")
    stagger_ms = int(text)
    if stagger_ms > maximum:
        raise ValueError(
            f"Request stagger cannot exceed {maximum:,} milliseconds.")
    return stagger_ms


def normalise_source_options(items):
    """Coerce backend catalogue entries into stable, unique UI records."""
    options = []
    seen_keys = set()
    for item in items:
        if isinstance(item, SourceUiOption):
            option = item
        elif isinstance(item, dict):
            option = SourceUiOption(
                key=str(
                    item.get("key", item.get("source_key", ""))).strip(),
                name=str(
                    item.get("name", item.get("title", ""))).strip(),
                word_count=item.get("word_count"),
                token_occurrence_count=item.get(
                    "token_occurrence_count"),
                section_count=item.get("section_count"),
                source_language_key=str(
                    item.get(
                        "source_language_key",
                        item.get("language_key", ""))).strip(),
                preset=bool(item.get("preset", False)))
        else:
            option = SourceUiOption(
                key=str(getattr(
                    item,
                    "key",
                    getattr(item, "source_key", ""))).strip(),
                name=str(getattr(
                    item,
                    "name",
                    getattr(item, "title", ""))).strip(),
                word_count=getattr(item, "word_count", None),
                token_occurrence_count=getattr(
                    item,
                    "token_occurrence_count",
                    None),
                section_count=getattr(item, "section_count", None),
                source_language_key=str(
                    getattr(
                        item,
                        "source_language_key",
                        getattr(item, "language_key", ""))).strip(),
                preset=bool(getattr(item, "preset", False)))
        if not option.key or not option.name or option.key in seen_keys:
            continue
        if (
                option.word_count is not None
                and (
                    isinstance(option.word_count, bool)
                    or not isinstance(option.word_count, int)
                    or option.word_count < 0)):
            continue
        if (
                option.token_occurrence_count is not None
                and (
                    isinstance(option.token_occurrence_count, bool)
                    or not isinstance(option.token_occurrence_count, int)
                    or option.token_occurrence_count < 0)):
            continue
        if (
                option.section_count is not None
                and (
                    isinstance(option.section_count, bool)
                    or not isinstance(option.section_count, int)
                    or option.section_count < 0)):
            continue
        seen_keys.add(option.key)
        options.append(option)
    return tuple(options)


def source_option_display_labels(options):
    """Return a stable, unique combobox label for every prepared-source key.

    Titles remain uncluttered when they are unique. If two prepared sources
    share a title, their immutable source keys disambiguate them without
    changing the title used for requests or output deck names.
    """
    options = tuple(options)
    title_counts = {}
    for option in options:
        title_counts[option.name] = title_counts.get(option.name, 0) + 1

    labels_by_key = {}
    used_labels = set()
    for option in sorted(options, key=lambda item: (item.key, item.name)):
        if title_counts[option.name] > 1:
            base_label = f"{option.name} · {option.key}"
        else:
            base_label = option.name
        label = base_label
        suffix = 2
        while label in used_labels:
            label = f"{base_label} · {suffix}"
            suffix += 1
        labels_by_key[option.key] = label
        used_labels.add(label)
    return labels_by_key


def format_source_estimate(estimate):
    """Return concise price and accounting text for mapping/object results."""
    if estimate is None:
        return (
            "Estimate unavailable",
            "Change an option to calculate an offline estimate.")

    def read(name, default=None):
        if isinstance(estimate, dict):
            return estimate.get(name, default)
        return getattr(estimate, name, default)

    low = read("estimated_cost_low_aud")
    high = read("estimated_cost_high_aud")
    exact = read("estimated_cost_aud")
    if low is None and high is None and exact is None:
        # Compatibility for estimators created before AUD aliases were added.
        aud_per_usd = float(read("usd_to_aud_rate", 1.0 / 0.6975))
        usd_low = read("estimated_cost_low_usd")
        usd_high = read("estimated_cost_high_usd")
        usd_exact = read("estimated_cost_usd")
        low = None if usd_low is None else float(usd_low) * aud_per_usd
        high = None if usd_high is None else float(usd_high) * aud_per_usd
        exact = None if usd_exact is None else float(usd_exact) * aud_per_usd
    if exact is not None:
        price = f"A${float(exact):,.2f} expected"
    elif low is not None or high is not None:
        low = float(low if low is not None else high)
        high = float(high if high is not None else low)
        price = f"A${low:,.2f}–A${high:,.2f}"
    else:
        price = "Estimate unavailable"

    request_count = read("request_count")
    candidate_count = read("candidate_count")
    input_tokens = read("input_tokens")
    output_tokens = read("output_tokens")
    details = []
    assumptions = read("assumptions", {})
    request_protocol = read("request_protocol")
    reasoning_effort = read("reasoning_effort")
    execution_mode = read("execution_mode")
    model = read("model")
    if isinstance(assumptions, dict):
        request_protocol = (
            request_protocol
            or assumptions.get("request_protocol"))
        reasoning_effort = (
            reasoning_effort
            or assumptions.get("reasoning_effort"))
        execution_mode = (
            execution_mode
            or assumptions.get("execution_mode"))
        model = model or assumptions.get("pricing", {}).get("model")
    mode_details = []
    if model:
        mode_details.append(str(model))
    if request_protocol:
        mode_details.append(
            SOURCE_PROTOCOL_ESTIMATE_LABELS.get(
                request_protocol,
                f"protocol {request_protocol}"))
    if reasoning_effort:
        mode_details.append(
            "no reasoning"
            if reasoning_effort == "none"
            else f"{reasoning_effort} reasoning")
    if execution_mode:
        mode_details.append(
            "Economy Batch"
            if execution_mode == "economy"
            else "Standard processing")
    if mode_details:
        details.append(" / ".join(mode_details))
    if low is not None or high is not None:
        range_low = float(low if low is not None else high)
        range_high = float(high if high is not None else low)
        details.append(
            f"modelled range A${range_low:,.2f}–A${range_high:,.2f}")
    prefix_limit = read("source_prefix_token_limit")
    prefix_token_count = read("source_prefix_token_count")
    prefix_unique_count = read("prefix_unique_candidate_count")
    if prefix_limit is not None:
        prefix_detail = (
            f"first {int(prefix_token_count):,} running-word occurrences"
            if prefix_token_count is not None
            else f"first {int(prefix_limit):,} running-word occurrences")
        if prefix_unique_count is not None:
            prefix_detail += (
                f" → {int(prefix_unique_count):,} unique candidates before "
                "learned-word exclusions")
        details.append(prefix_detail)
    if candidate_count is not None:
        details.append(f"{int(candidate_count):,} new words")
    if request_count is not None:
        details.append(f"{int(request_count):,} requests")
    if input_tokens is not None:
        details.append(f"{int(input_tokens):,} estimated input tokens")
    if output_tokens is not None:
        details.append(f"{int(output_tokens):,} estimated output tokens")
    cached_input = read("estimated_cached_input_tokens")
    if cached_input:
        details.append(
            f"{int(cached_input):,} input tokens expected at cache-read rate")
    cache_write_input = read("estimated_cache_write_input_tokens")
    if cache_write_input:
        details.append(
            f"{int(cache_write_input):,} input tokens expected at "
            "cache-write rate")
    translation_memory_hits = read("translation_memory_hit_count")
    if translation_memory_hits:
        details.append(
            f"{int(translation_memory_hits):,} exact source translations "
            "reused locally")
    if read("automatic_repair"):
        reserve_aud = (
            assumptions.get("automatic_repair_reserve_aud")
            if isinstance(assumptions, dict)
            else None)
        maximum_requests = (
            assumptions.get("automatic_repair_max_requests")
            if isinstance(assumptions, dict)
            else None)
        details.append(
            "automatic repair enabled: exact translation reuse and up to "
            "3 additional paid micro-repair calls per failed request")
        if reserve_aud is not None:
            details.append(
                "worst-case optional repair reserve "
                f"A${float(reserve_aud):,.2f}"
                + (
                    f" across at most {int(maximum_requests):,} calls"
                    if maximum_requests is not None
                    else "")
                + " (excluded from headline)")
    pricing_label = read("pricing_label")
    if pricing_label:
        details.append(str(pricing_label))
    largest_input = read("largest_request_input_tokens")
    largest_output = read("largest_request_output_tokens")
    if largest_input is not None and largest_output is not None:
        details.append(
            "largest request ≈ "
            f"{int(largest_input):,} input / "
            f"{int(largest_output):,} output tokens")
    if details and execution_mode != "economy":
        details.append("automatic transient retries are not included")
    if (
            isinstance(assumptions, dict)
            and assumptions.get("web_search_enabled")):
        details.append(
            "web-search range assumes zero to one search per request")
    if isinstance(assumptions, dict) and assumptions.get("aud_exchange_rate"):
        details.append(str(assumptions["aud_exchange_rate"]))
    detail = " · ".join(details)
    if not detail:
        detail = str(read(
            "description",
            "Offline estimate based on the selected card setup."))
    warning = read("request_limit_warning")
    if warning:
        detail = f"Reduce the chunk size — {warning} {detail}"
    return price, detail


def language_selector_width():
    """Size language menus for their longest label with comfortable padding."""
    return max(
        24,
        max(
            len(language.name)
            for language in pipeline_store.list_languages())
        + 4)


def wheel_scroll_amount(event):
    """Translate Windows/macOS/X11 wheel events into one scroll unit."""
    if getattr(event, "num", None) == 4:
        if int(getattr(event, "state", 0) or 0) & 1:
            return 0
        return -1
    if getattr(event, "num", None) == 5:
        return 1
    delta = getattr(event, "delta", 0)
    return -1 if delta > 0 else 1 if delta < 0 else 0


class RoundedPanel(tk.Canvas):
    """A responsive rounded surface containing ordinary Tk widgets."""

    def __init__(
            self,
            parent,
            *,
            fill,
            outline,
            background,
            radius=16,
            inset=10):
        super().__init__(
            parent,
            background=background,
            borderwidth=0,
            highlightthickness=0,
            takefocus=False)
        self._fill = fill
        self._outline = outline
        self._radius = radius
        self._inset = inset
        self._shape = self.create_polygon(
            0, 0, 1, 0, 1, 1, 0, 1,
            fill=fill,
            outline=outline,
            width=1,
            smooth=True,
            splinesteps=24)
        self.interior = tk.Frame(
            self,
            background=fill,
            borderwidth=0)
        self._window = self.create_window(
            inset,
            inset,
            window=self.interior,
            anchor="nw")
        self.interior.bind(
            "<Configure>",
            self._fit_to_contents,
            add="+")
        self.bind(
            "<Configure>",
            self._resize_surface,
            add="+")

    def _rounded_points(self, width, height):
        radius = min(
            self._radius,
            max(1, width // 2),
            max(1, height // 2))
        return (
            radius, 1,
            width - radius, 1,
            width - 1, 1,
            width - 1, radius,
            width - 1, height - radius,
            width - 1, height - 1,
            width - radius, height - 1,
            radius, height - 1,
            1, height - 1,
            1, height - radius,
            1, radius,
            1, 1)

    def _fit_to_contents(self, _event=None):
        self.configure(
            width=max(
                1,
                self.interior.winfo_reqwidth() + 2 * self._inset),
            height=max(
                1,
                self.interior.winfo_reqheight() + 2 * self._inset))

    def _resize_surface(self, event):
        width = max(2, event.width)
        height = max(2, event.height)
        self.coords(
            self._shape,
            *self._rounded_points(width, height))
        self.itemconfigure(
            self._window,
            width=max(1, width - 2 * self._inset),
            height=max(1, height - 2 * self._inset))


class RoundedScrollbar(tk.Canvas):
    """A continuous scrollbar with a rounded, non-tiled thumb."""

    def __init__(
            self,
            parent,
            *,
            command,
            background,
            thumb="#9AA9A5",
            active="#176C64",
            width=14):
        super().__init__(
            parent,
            width=width,
            # A bare Tk Canvas requests roughly 276 px of height. Vertical
            # scrollbars are always stretched by their geometry manager, so
            # that default only forces otherwise compact readers and tables
            # to become needlessly tall.
            height=1,
            background=background,
            borderwidth=0,
            highlightthickness=0,
            takefocus=False)
        self._command = command
        self._thumb_colour = thumb
        self._active_colour = active
        self._first = 0.0
        self._last = 1.0
        self._drag_offset = 0.0
        self._thumb_top = 0.0
        self._thumb_bottom = 0.0
        self._thumb = self.create_polygon(
            0, 0, 1, 0, 1, 1, 0, 1,
            fill=thumb,
            outline="",
            smooth=True,
            splinesteps=24)
        self.bind("<Configure>", self._redraw, add="+")
        self.bind("<Button-1>", self._press, add="+")
        self.bind("<B1-Motion>", self._drag, add="+")
        self.bind("<ButtonRelease-1>", self._release, add="+")
        self.bind("<Enter>", self._activate, add="+")
        self.bind("<Leave>", self._deactivate, add="+")

    @staticmethod
    def _pill_points(left, top, right, bottom):
        radius = min((right - left) / 2, (bottom - top) / 2)
        return (
            left + radius, top,
            right - radius, top,
            right, top,
            right, top + radius,
            right, bottom - radius,
            right, bottom,
            right - radius, bottom,
            left + radius, bottom,
            left, bottom,
            left, bottom - radius,
            left, top + radius,
            left, top)

    def set(self, first, last):
        self._first = max(0.0, min(1.0, float(first)))
        self._last = max(self._first, min(1.0, float(last)))
        self._redraw()

    def _redraw(self, _event=None):
        height = max(1, self.winfo_height())
        width = max(1, self.winfo_width())
        visible = max(0.0, self._last - self._first)
        thumb_height = min(height, max(28.0, height * visible))
        travel = max(0.0, height - thumb_height)
        denominator = max(0.000001, 1.0 - visible)
        top = travel * self._first / denominator
        left = 3.0
        right = max(left + 2.0, width - 3.0)
        self._thumb_top = top
        self._thumb_bottom = top + thumb_height
        self.coords(
            self._thumb,
            *self._pill_points(
                left,
                top,
                right,
                top + thumb_height))
        self.itemconfigure(
            self._thumb,
            state="hidden" if visible >= 0.999999 else "normal")

    def _press(self, event):
        if self._thumb_top <= event.y <= self._thumb_bottom:
            self._drag_offset = event.y - self._thumb_top
        else:
            self._drag_offset = (
                self._thumb_bottom - self._thumb_top) / 2
            self._move_to(event.y - self._drag_offset)
        return "break"

    def _drag(self, event):
        self._move_to(event.y - self._drag_offset)
        return "break"

    def _move_to(self, top):
        height = max(1.0, float(self.winfo_height()))
        thumb_height = self._thumb_bottom - self._thumb_top
        travel = max(1.0, height - thumb_height)
        fraction = max(0.0, min(1.0, top / travel))
        self._command("moveto", fraction)

    def _release(self, _event):
        return "break"

    def _activate(self, _event=None):
        self.itemconfigure(
            self._thumb,
            fill=self._active_colour)

    def _deactivate(self, _event=None):
        self.itemconfigure(
            self._thumb,
            fill=self._thumb_colour)


class ScrollableFrame(ttk.Frame):
    """A full-width vertical viewport for settings that may grow."""

    def __init__(
            self,
            parent,
            *,
            background,
            frame_style):
        super().__init__(
            parent,
            style=frame_style)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self,
            background=background,
            borderwidth=0,
            highlightthickness=0,
            takefocus=False)
        self.scrollbar = RoundedScrollbar(
            self,
            command=self.canvas.yview,
            background=background)
        self.canvas.configure(
            yscrollcommand=self.scrollbar.set)
        self.canvas.grid(
            row=0,
            column=0,
            sticky="nsew")
        self.scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
            padx=(7, 0))

        self.content = ttk.Frame(
            self.canvas,
            padding=(20, 18, 20, 32),
            style=frame_style)
        self._content_window = self.canvas.create_window(
            0,
            0,
            window=self.content,
            anchor="nw")
        self.content.bind(
            "<Configure>",
            self._content_changed,
            add="+")
        self.canvas.bind(
            "<Configure>",
            self._viewport_changed,
            add="+")

    def _content_changed(self, _event=None):
        self.canvas.configure(
            scrollregion=self.canvas.bbox("all"))

    def _viewport_changed(self, event):
        self.canvas.itemconfigure(
            self._content_window,
            width=max(1, event.width))

    def _wheel_scroll(self, event):
        amount = wheel_scroll_amount(event)
        if amount:
            self.canvas.yview_scroll(amount, "units")
            return "break"
        return None

    def bind_mousewheel_tree(self, *, preserve_scrollable_children=False):
        """Route wheel events from child controls to this viewport.

        A page containing its own large trees or text readers can preserve
        those widgets' native scrolling while the surrounding chrome still
        scrolls the outer page.
        """
        stack = [self.canvas, self.content]
        while stack:
            widget = stack.pop()
            preserve_widget = (
                preserve_scrollable_children
                and widget not in {self.canvas, self.content}
                and isinstance(
                    widget,
                    (tk.Text, tk.Listbox, ttk.Treeview)))
            if not preserve_widget:
                widget.bind(
                    "<MouseWheel>",
                    self._wheel_scroll)
                widget.bind(
                    "<Button-4>",
                    self._wheel_scroll)
                widget.bind(
                    "<Button-5>",
                    self._wheel_scroll)
            stack.extend(widget.winfo_children())


class _LegacyPipelineEditor:
    CARD_DESCRIPTIONS = {
        "Context sentence → meaning": (
            "A varied example sentence on the front; pronunciation and "
            "meaning on the back."),
        "Word → meaning": (
            "The word or expression on the front and its meaning on the "
            "back."),
        "Meaning → word": (
            "The meaning on the front and the word or expression on the "
            "back."),
        "Context sentence → English definition": (
            "A native-language example sentence on the front; pronunciation "
            "and an English definition on the back."),
        "Word → English definition": (
            "The word or expression on the front and its English definition "
            "on the back."),
        "English definition → word": (
            "The English definition on the front and the word or expression "
            "on the back."),
        "Context sentence → native definition": (
            "A native-language example sentence on the front; pronunciation "
            "and a native-language definition on the back."),
        "Word → native definition": (
            "The word or expression on the front and its native-language "
            "definition on the back."),
        "Native definition → word": (
            "The native-language definition on the front and the word or "
            "expression on the back."),
    }

    def __init__(self, app, parent, config):
        self.app = app
        self.config = config
        self.frame = RoundedPanel(
            parent,
            fill=app.PANEL_BACKGROUND,
            outline=app.LINE,
            background=app.WINDOW_BACKGROUND,
            radius=26,
            inset=18)
        self.frame.grid(row=0, column=0, sticky="nsew")
        panel = self.frame.interior
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        self.language_tabs = {}
        self.language_by_tab = {}
        self.language_scroll_frames = {}
        self.target_deck_variables = {}
        self.target_deck_boxes = {}
        self.shared_deck_labels = {}
        self.separate_target_deck_variables = {}
        self.card_output_variables = {}
        self.card_target_deck_variables = {}
        self.card_target_deck_boxes = {}
        self.card_target_deck_containers = {}
        self.card_target_deck_shared_hints = {}
        self.card_output_traces = []
        self.variable_traces = []

        ttk.Label(
            panel,
            text="CHOOSE A LANGUAGE",
            style="Section.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                pady=(0, 10))

        self.language_notebook = ttk.Notebook(
            panel,
            padding=(0, 0))
        self.language_notebook.grid(
            row=1,
            column=0,
            sticky="nsew")

        for language in pipeline_store.list_settings_languages():
            self._add_language_tab(language, config)

        active_settings_language_key = pipeline_store.get_language(
            config.language_key).model_language_key
        active_tab = self.language_tabs.get(
            active_settings_language_key)
        if active_tab is not None:
            self.language_notebook.select(active_tab)
        self.last_active_language_key = active_settings_language_key
        self.language_notebook.bind(
            "<<NotebookTabChanged>>",
            self._language_changed)

        self._update_deck_mode_visibility()

    def _deck_mode_changed(self, language_key):
        self._update_deck_mode_visibility(language_key)
        self.app._update_pipeline_status()

    def _update_deck_mode_visibility(self, language_key=None):
        language_keys = (
            (language_key,)
            if language_key is not None
            else tuple(self.card_target_deck_containers))
        for current_language_key in language_keys:
            separate = not self.separate_target_deck_variables[
                current_language_key].get()
            shared_label = self.shared_deck_labels[current_language_key]
            shared_box = self.target_deck_boxes[current_language_key]
            if separate:
                shared_label.grid_remove()
                shared_box.grid_remove()
            else:
                shared_label.grid()
                shared_box.grid()

            containers = self.card_target_deck_containers[
                current_language_key]
            shared_hints = self.card_target_deck_shared_hints[
                current_language_key]
            selected_variables = self.card_output_variables[
                current_language_key]
            for card_type_key, container in containers.items():
                if separate:
                    shared_hints[card_type_key].place_forget()
                    state = (
                        "normal"
                        if selected_variables[card_type_key].get()
                        else "disabled")
                    self.card_target_deck_boxes[
                        current_language_key][card_type_key].configure(
                            state=state)
                else:
                    self.card_target_deck_boxes[
                        current_language_key][card_type_key].configure(
                            state="disabled")
                    shared_hints[card_type_key].place(
                        relx=0,
                        rely=0,
                        relwidth=1,
                        relheight=1)

    def _add_language_tab(self, language, config):
        tab = ttk.Frame(
            self.language_notebook,
            style="Panel.TFrame")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        self.language_notebook.add(tab, text=language.name)
        self.language_tabs[language.key] = tab
        self.language_by_tab[str(tab)] = language
        scroll_frame = ScrollableFrame(
            tab,
            background=self.app.PANEL_BACKGROUND,
            frame_style="Panel.TFrame")
        scroll_frame.grid(
            row=0,
            column=0,
            sticky="nsew")
        self.language_scroll_frames[language.key] = scroll_frame
        content = scroll_frame.content
        settings = pipeline_store.get_language_settings(
            config,
            language.key)

        definition_groups = [(
            "ENGLISH-DEFINITION CARDS",
            tuple(
                card_type
                for card_type in language.card_types
                if "native definition" not in card_type[1].lower()),
        )]
        native_card_types = tuple(
            card_type
            for card_type in language.card_types
            if "native definition" in card_type[1].lower())
        if native_card_types:
            definition_groups.append((
                f"{language.name.upper()}-DEFINITION CARDS",
                native_card_types,
            ))
        content.columnconfigure(0, weight=1)

        ttk.Label(
            content,
            text="Cards to create",
            style="Section.TLabel").grid(
                row=0,
                column=0,
                sticky="w")
        ttk.Label(
            content,
            text=(
                "Select one or more outputs. They will share a single "
                "OpenAI response."),
            style="Muted.TLabel").grid(
                row=1,
                column=0,
                sticky="w",
                pady=(3, 14))

        destination = ttk.Frame(
            content,
            style="Panel.TFrame")
        destination.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 12))
        destination.columnconfigure(1, weight=1)
        separate_target_decks = tk.BooleanVar(
            value=not settings.separate_target_decks)
        self.separate_target_deck_variables[
            language.key] = separate_target_decks
        split_decks_button = ttk.Checkbutton(
            destination,
            text="Send each card type to the same Anki deck",
            variable=separate_target_decks,
            command=lambda key=language.key: (
                self._deck_mode_changed(key)),
            style="Panel.TCheckbutton")
        split_decks_button.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 7))

        shared_deck_label = ttk.Label(
            destination,
            text="TARGET ANKI DECK",
            style="FieldLabel.TLabel")
        shared_deck_label.grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 8))
        self.shared_deck_labels[language.key] = shared_deck_label

        target_deck = tk.StringVar(
            value=settings.target_deck)
        self.target_deck_variables[language.key] = target_deck
        target_deck_box = ttk.Combobox(
            destination,
            textvariable=target_deck,
            values=self.app.deck_options,
            width=42,
            state="normal",
            style="App.TCombobox")
        target_deck_box.grid(
            row=1,
            column=1,
            sticky="ew")
        target_deck_box.bind(
            "<FocusOut>",
            self.app._clear_entry_selection,
            add="+")
        target_deck_box.bind(
            "<Button-1>",
            self.app._deck_selector_opened,
            add="+")
        self.target_deck_boxes[language.key] = target_deck_box

        ttk.Label(
            destination,
            textvariable=self.app.deck_status,
            style="Muted.TLabel").grid(
                row=2,
                column=0,
                columnspan=2,
                sticky="w",
                pady=(5, 0))
        self.variable_traces.extend((
            (
                target_deck,
                target_deck.trace_add(
                    "write",
                    self.app.schedule_pipeline_save)),
            (
                separate_target_decks,
                separate_target_decks.trace_add(
                    "write",
                    self.app.schedule_pipeline_save)),
        ))

        selected_keys = set(settings.card_type_keys)
        saved_target_decks = dict(
            settings.card_type_target_decks)
        variables = {}
        target_deck_variables = {}
        target_deck_boxes = {}
        target_deck_containers = {}
        target_deck_shared_hints = {}
        for column_number, (group_label, card_types) in enumerate(
                definition_groups):
            group = ttk.Frame(
                content,
                style="Panel.TFrame")
            group.grid(
                row=3 + column_number,
                column=0,
                sticky="ew",
                pady=(
                    (0, 10)
                    if column_number < len(definition_groups) - 1
                    else 0))
            group.columnconfigure(0, weight=1)
            ttk.Label(
                group,
                text=group_label,
                style="FieldLabel.TLabel").grid(
                    row=0,
                    column=0,
                    sticky="w",
                    pady=(0, 7))

            for row_number, (card_type_key, label) in enumerate(
                    card_types,
                    start=1):
                option_surface = RoundedPanel(
                    group,
                    fill=self.app.PALE,
                    outline="#C8DDD8",
                    background=self.app.PANEL_BACKGROUND,
                    radius=18,
                    inset=7)
                option_surface.grid(
                    row=row_number,
                    column=0,
                    sticky="ew",
                    pady=(0, 7))
                option = option_surface.interior
                option.columnconfigure(0, weight=5)
                option.columnconfigure(1, weight=2)

                variable = tk.BooleanVar(
                    value=card_type_key in selected_keys)
                variables[card_type_key] = variable
                ttk.Checkbutton(
                    option,
                    text=label,
                    variable=variable,
                    style="CardOption.TCheckbutton").grid(
                        row=0,
                        column=0,
                        sticky="w")

                target_deck_variable = tk.StringVar(
                    value=saved_target_decks.get(
                        card_type_key,
                        settings.target_deck))
                target_deck_variables[card_type_key] = (
                    target_deck_variable)
                deck_container = ttk.Frame(
                    option,
                    style="Pale.TFrame")
                deck_container.grid(
                    row=0,
                    column=1,
                    sticky="ew",
                    padx=(12, 0))
                deck_container.columnconfigure(0, weight=1)
                target_deck_box = ttk.Combobox(
                    deck_container,
                    textvariable=target_deck_variable,
                    values=self.app.deck_options,
                    width=20,
                    state="normal",
                    style="App.TCombobox")
                target_deck_box.grid(
                    row=0,
                    column=0,
                    sticky="ew")
                shared_deck_hint = ttk.Label(
                    deck_container,
                    text="Uses shared language deck",
                    anchor="center",
                    style="CardDescription.TLabel")
                shared_deck_hint.place(
                    relx=0,
                    rely=0,
                    relwidth=1,
                    relheight=1)
                target_deck_box.bind(
                    "<FocusOut>",
                    self.app._clear_entry_selection,
                    add="+")
                target_deck_box.bind(
                    "<Button-1>",
                    self.app._deck_selector_opened,
                    add="+")
                target_deck_boxes[card_type_key] = target_deck_box
                target_deck_containers[card_type_key] = deck_container
                target_deck_shared_hints[
                    card_type_key] = shared_deck_hint
                deck_trace_id = target_deck_variable.trace_add(
                    "write",
                    self.app.schedule_pipeline_save)
                self.variable_traces.append(
                    (target_deck_variable, deck_trace_id))

                trace_id = variable.trace_add(
                    "write",
                    self._card_selection_changed)
                self.card_output_traces.append((variable, trace_id))

        self.card_output_variables[language.key] = variables
        self.card_target_deck_variables[
            language.key] = target_deck_variables
        self.card_target_deck_boxes[
            language.key] = target_deck_boxes
        self.card_target_deck_containers[
            language.key] = target_deck_containers
        self.card_target_deck_shared_hints[
            language.key] = target_deck_shared_hints
        scroll_frame.bind_mousewheel_tree()

    def _language_changed(self, _event=None):
        language = self.get_active_language()
        if language.key == self.last_active_language_key:
            return
        self.last_active_language_key = language.key
        self._update_deck_mode_visibility()
        self.app.schedule_pipeline_save()
        self.app._update_pipeline_status()

    def _card_selection_changed(self, *_args):
        self._update_deck_mode_visibility()
        self.app.schedule_pipeline_save()
        self.app._update_pipeline_status()

    def get_active_language(self):
        selected_tab = self.language_notebook.select()
        language = self.language_by_tab.get(selected_tab)
        if language is None:
            raise ValueError("Select a language.")
        return language

    def to_config(self):
        active_language = self.app.get_generation_language()
        language_settings = []
        for language in pipeline_store.list_settings_languages():
            variables = self.card_output_variables[language.key]
            card_type_keys = tuple(
                card_type_key
                for card_type_key, _label in language.card_types
                if variables[card_type_key].get())
            target_deck_variables = (
                self.card_target_deck_variables[language.key])
            card_type_target_decks = tuple(
                (
                    card_type_key,
                    target_deck_variables[card_type_key].get().strip(),
                )
                for card_type_key in card_type_keys)
            language_settings.append(
                pipeline_store.LanguageSettings(
                    language_key=language.key,
                    card_type_keys=card_type_keys,
                    target_deck=self.target_deck_variables[
                        language.key].get().strip(),
                    separate_target_decks=(
                        not self.separate_target_deck_variables[
                            language.key].get()),
                    card_type_target_decks=card_type_target_decks))
        active_settings = next(
            settings
            for settings in language_settings
            if settings.language_key == active_language.key)

        return replace(
            self.config,
            language_key=active_language.key,
            card_type_keys=active_settings.card_type_keys,
            target_deck=active_settings.target_deck,
            separate_target_decks=(
                active_settings.separate_target_decks),
            card_type_target_decks=(
                active_settings.card_type_target_decks),
            language_settings=tuple(language_settings))

    def update_deck_options(self, deck_options):
        for box in self.target_deck_boxes.values():
            box.configure(values=deck_options)
        for boxes in self.card_target_deck_boxes.values():
            for box in boxes.values():
                box.configure(values=deck_options)

    def destroy(self):
        for variable, trace_id in self.card_output_traces:
            variable.trace_remove("write", trace_id)
        for variable, trace_id in self.variable_traces:
            variable.trace_remove("write", trace_id)
        self.frame.destroy()


class PipelineEditor:
    """Editor for three card directions and configurable meaning fields."""

    def __init__(self, app, parent, config):
        self.app = app
        self.config = config
        self.variable_traces = []
        self.language_tabs = {}
        self.language_by_tab = {}
        self.target_deck_variables = {}
        self.target_deck_boxes = {}
        self.separate_target_deck_variables = {}
        self.share_field_settings_variables = {}
        self.learned_filter_enabled_variables = {}
        self.learned_filter_summary_variables = {}
        self.make_items_for_characters_variables = {}
        self.direction_enabled_variables = {}
        self.direction_enhanced_variables = {}
        self.direction_enhanced_checks = {}
        self.sentence_translation_variables = {}
        self.sentence_translation_checks = {}
        self.card_output_variables = self.direction_enabled_variables
        self.direction_target_deck_variables = {}
        self.direction_target_deck_boxes = {}
        self.direction_surfaces = {}
        self.shared_field_enabled_variables = {}
        self.shared_field_language_variables = {}
        self.shared_field_language_boxes = {}
        self.shared_field_containers = {}
        self.per_card_field_enabled_variables = {}
        self.per_card_field_language_variables = {}
        self.per_card_field_language_boxes = {}
        self.per_card_field_containers = {}
        self.built_language_keys = set()

        self.frame = RoundedPanel(
            parent,
            fill=app.PANEL_BACKGROUND,
            outline=app.LINE,
            background=app.WINDOW_BACKGROUND,
            radius=26,
            inset=18)
        self.frame.grid(row=0, column=0, sticky="ew")
        panel = self.frame.interior
        panel.columnconfigure(0, weight=1)

        ttk.Label(
            panel,
            text="CHOOSE A LANGUAGE",
            style="Section.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                pady=(0, 10))
        self.language_notebook = ttk.Notebook(
            panel,
            padding=(0, 0))
        self.language_notebook.grid(
            row=1,
            column=0,
            sticky="nsew")

        for language in pipeline_store.list_settings_languages():
            self._register_language_tab(language)
        active_settings_language_key = (
            self.app.get_card_setup_generation_language().model_language_key)
        active_tab = self.language_tabs.get(
            active_settings_language_key)
        if active_tab is not None:
            self.language_notebook.select(active_tab)
            self._ensure_language_tab_built(
                active_settings_language_key)
        self.last_active_language_key = active_settings_language_key
        self.language_notebook.bind(
            "<<NotebookTabChanged>>",
            self._language_changed)
        if active_settings_language_key in self.built_language_keys:
            self._update_visibility(active_settings_language_key)

    def _trace(self, variable, callback=None):
        trace_id = variable.trace_add(
            "write",
            callback or self.app.schedule_pipeline_save)
        self.variable_traces.append((variable, trace_id))

    def _concrete_source_language_key(self, settings_language_key):
        """Resolve a shared Card Setup tab to the active language variant."""
        active_language = self.app.get_card_setup_generation_language()
        if active_language.model_language_key == settings_language_key:
            return active_language.key
        return settings_language_key

    def _language_values(self, source_language_key, field_key):
        return tuple(
            language.name
            for language in pipeline_store.list_response_languages()
            if not (
                field_key == "translation"
                and not pipeline_store.translation_target_allowed(
                    source_language_key,
                    language.key)))

    def _default_field_language_key(
            self,
            source_language_key,
            field_key):
        allowed = tuple(
            language.key
            for language in pipeline_store.list_response_languages()
            if (
                field_key != "translation"
                or pipeline_store.translation_target_allowed(
                    source_language_key,
                    language.key)))
        if "english" in allowed:
            return "english"
        if not allowed:
            raise ValueError(
                "No valid response language is available for this field.")
        return allowed[0]

    def _language_key_from_name(self, language_name):
        for language in pipeline_store.list_response_languages():
            if language.name == language_name:
                return language.key
        raise ValueError(
            f"Unknown target language: {language_name}")

    def _create_field_grid(
            self,
            parent,
            language,
            selected_fields,
            remembered_fields=()):
        remembered_by_key = {
            field.field_key: field.target_language_key
            for field in remembered_fields}
        selected_by_key = {
            field.field_key: field.target_language_key
            for field in selected_fields}
        enabled_variables = {}
        language_variables = {}
        language_boxes = {}
        grid = ttk.Frame(parent, style="Pale.TFrame")
        grid.columnconfigure(1, weight=1)

        for row, field in enumerate(
                pipeline_store.list_field_options()):
            concrete_source_key = self._concrete_source_language_key(
                language.key)
            target_values = self._language_values(
                concrete_source_key,
                field.key)
            selected_target_key = selected_by_key.get(
                field.key,
                remembered_by_key.get(field.key))
            if (
                    selected_target_key is not None
                    and field.key == "translation"
                    and not pipeline_store.translation_target_allowed(
                        concrete_source_key,
                        selected_target_key)):
                selected_target_key = None
            if selected_target_key is None:
                selected_target_key = self._default_field_language_key(
                    concrete_source_key,
                    field.key)
            selected_target_name = pipeline_store.get_language(
                selected_target_key).name
            enabled = tk.BooleanVar(
                value=field.key in selected_by_key)
            target = tk.StringVar(value=selected_target_name)
            enabled_variables[field.key] = enabled
            language_variables[field.key] = target

            ttk.Checkbutton(
                grid,
                text=field.name,
                variable=enabled,
                command=lambda key=language.key: (
                    self._field_controls_changed(key)),
                style="CardOption.TCheckbutton").grid(
                    row=row,
                    column=0,
                    sticky="w",
                    padx=(0, 12),
                    pady=3)
            box = ttk.Combobox(
                grid,
                textvariable=target,
                values=target_values,
                width=language_selector_width(),
                state="readonly",
                style="App.TCombobox")
            box.grid(
                row=row,
                column=1,
                sticky="ew",
                pady=3)
            box.bind(
                "<<ComboboxSelected>>",
                lambda _event, key=language.key: (
                    self._field_controls_changed(key)),
                add="+")
            box.bind(
                "<FocusOut>",
                self.app._clear_entry_selection,
                add="+")
            language_boxes[field.key] = box
            self._trace(enabled)
            self._trace(target)
        return (
            grid,
            enabled_variables,
            language_variables,
            language_boxes)

    def _register_language_tab(self, language):
        tab = ttk.Frame(
            self.language_notebook,
            style="Panel.TFrame")
        tab.columnconfigure(0, weight=1)
        self.language_notebook.add(tab, text=language.name)
        self.language_tabs[language.key] = tab
        self.language_by_tab[str(tab)] = language

    def _ensure_language_tab_built(self, language_key):
        if language_key in self.built_language_keys:
            return False
        language = next(
            (
                item
                for item in pipeline_store.list_settings_languages()
                if item.key == language_key
            ),
            None)
        if language is None:
            raise ValueError(
                f"Unknown Card Setup language: {language_key}")
        self._build_language_tab(language)
        self.built_language_keys.add(language_key)
        return True

    def _build_language_tab(self, language):
        tab = self.language_tabs[language.key]
        content = ttk.Frame(
            tab,
            padding=(20, 18, 20, 32),
            style="Panel.TFrame")
        content.grid(row=0, column=0, sticky="ew")
        content.columnconfigure(0, weight=1)
        settings = pipeline_store.get_language_settings(
            self.config,
            language.key)

        ttk.Label(
            content,
            text="Cards to create",
            style="Section.TLabel").grid(
                row=0,
                column=0,
                sticky="w")
        ttk.Label(
            content,
            text=(
                "Choose card directions and exactly which information "
                "OpenAI should return. Unselected fields are omitted."),
            style="Muted.TLabel",
            wraplength=1000).grid(
                row=1,
                column=0,
                sticky="w",
                pady=(3, 12))

        controls = ttk.Frame(content, style="Panel.TFrame")
        controls.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 10))
        controls.columnconfigure(1, weight=1)

        same_deck = tk.BooleanVar(
            value=not settings.separate_target_decks)
        # Retain the existing attribute name for internal/test compatibility;
        # its UI value now expresses the positive, user-facing choice.
        self.separate_target_deck_variables[language.key] = same_deck
        ttk.Checkbutton(
            controls,
            text="Send each card type to the same Anki deck",
            variable=same_deck,
            command=lambda key=language.key: (
                self._layout_mode_changed(key)),
            style="Panel.TCheckbutton").grid(
                row=0,
                column=0,
                columnspan=2,
                sticky="w")

        share_fields = tk.BooleanVar(
            value=settings.share_field_settings)
        self.share_field_settings_variables[language.key] = share_fields
        ttk.Checkbutton(
            controls,
            text=(
                "Use the same selected fields and response languages "
                "for all card types"),
            variable=share_fields,
            command=lambda key=language.key: (
                self._layout_mode_changed(key)),
            style="Panel.TCheckbutton").grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="w",
                pady=(4, 10))

        ttk.Label(
            controls,
            text="TARGET ANKI DECK",
            style="FieldLabel.TLabel").grid(
                row=2,
                column=0,
                sticky="w",
                padx=(0, 10))
        target_deck = tk.StringVar(value=settings.target_deck)
        self.target_deck_variables[language.key] = target_deck
        target_box = ttk.Combobox(
            controls,
            textvariable=target_deck,
            values=self.app.deck_options,
            width=42,
            state="normal",
            style="App.TCombobox")
        target_box.grid(row=2, column=1, sticky="ew")
        target_box.bind(
            "<FocusOut>",
            self.app._clear_entry_selection,
            add="+")
        target_box.bind(
            "<Button-1>",
            self.app._deck_selector_opened,
            add="+")
        self.target_deck_boxes[language.key] = target_box
        ttk.Label(
            controls,
            textvariable=self.app.deck_status,
            style="Muted.TLabel").grid(
                row=3,
                column=0,
                columnspan=2,
                sticky="w",
                pady=(5, 0))
        learned_settings = self.app._language_filter(language.key)
        learned_enabled = tk.BooleanVar(
            value=learned_settings.enabled)
        learned_summary = tk.StringVar()
        self.learned_filter_enabled_variables[
            language.key] = learned_enabled
        self.learned_filter_summary_variables[
            language.key] = learned_summary
        learned_controls = ttk.Frame(
            controls,
            style="Panel.TFrame")
        learned_controls.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(12, 0))
        learned_controls.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            learned_controls,
            text="Omit words already learned in Anki",
            variable=learned_enabled,
            command=lambda key=language.key: (
                self._learned_filter_changed(key)),
            style="Panel.TCheckbutton").grid(
                row=0,
                column=0,
                sticky="w")
        ttk.Label(
            learned_controls,
            textvariable=learned_summary,
            style="Muted.TLabel",
            wraplength=520).grid(
                row=0,
                column=1,
                sticky="e",
                padx=(12, 0))
        ttk.Button(
            learned_controls,
            text="Choose decks and fields…",
            command=lambda key=language.key: (
                self.app._open_learned_filter_dialog(key)),
            style="CompactSecondary.TButton",
            cursor="hand2").grid(
                row=0,
                column=2,
                sticky="e",
                padx=(12, 0))
        if language.key in {"classical_chinese", "japanese"}:
            make_character_items = tk.BooleanVar(
                value=settings.make_items_for_characters)
            self.make_items_for_characters_variables[
                language.key] = make_character_items
            ttk.Checkbutton(
                learned_controls,
                text="Make items for characters in multi-character words",
                variable=make_character_items,
                command=self.app.schedule_pipeline_save,
                style="Panel.TCheckbutton").grid(
                    row=1,
                    column=0,
                    columnspan=3,
                    sticky="w",
                    pady=(8, 0))
            if language.key == "japanese":
                ttk.Label(
                    learned_controls,
                    text=(
                        "Note: Japanese-specific logic is yet to be "
                        "decided."),
                    style="Muted.TLabel").grid(
                        row=2,
                        column=0,
                        columnspan=3,
                        sticky="w",
                        padx=(24, 0),
                        pady=(3, 0))
            self._trace(make_character_items)
        self._trace(same_deck)
        self._trace(share_fields)
        self._trace(target_deck)
        self._update_learned_filter_summary(language.key)

        shared_surface = RoundedPanel(
            content,
            fill=self.app.PALE,
            outline="#C8DDD8",
            background=self.app.PANEL_BACKGROUND,
            radius=18,
            inset=10)
        shared_surface.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(0, 10))
        shared_content = shared_surface.interior
        shared_content.columnconfigure(0, weight=1)
        ttk.Label(
            shared_content,
            text="SHARED DEFINITION CONTENT",
            style="PaleField.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                pady=(0, 6))
        (
            shared_grid,
            shared_enabled,
            shared_languages,
            shared_boxes,
        ) = self._create_field_grid(
            shared_content,
            language,
            settings.shared_fields,
            settings.shared_field_languages)
        shared_grid.grid(row=1, column=0, sticky="ew")
        self.shared_field_containers[language.key] = shared_surface
        self.shared_field_enabled_variables[
            language.key] = shared_enabled
        self.shared_field_language_variables[
            language.key] = shared_languages
        self.shared_field_language_boxes[
            language.key] = shared_boxes

        cards_by_direction = {
            card.direction_key: card
            for card in settings.cards}
        enabled_variables = {}
        enhanced_variables = {}
        enhanced_checks = {}
        include_sentence_translations = tk.BooleanVar(
            value=settings.include_sentence_translations)
        sentence_translation_check = None
        deck_variables = {}
        deck_boxes = {}
        field_enabled_by_direction = {}
        field_languages_by_direction = {}
        field_boxes_by_direction = {}
        field_containers = {}
        direction_surfaces = {}

        for row, direction in enumerate(
                pipeline_store.list_directions(),
                start=4):
            card = cards_by_direction[direction.key]
            surface = RoundedPanel(
                content,
                fill=self.app.PALE,
                outline="#C8DDD8",
                background=self.app.PANEL_BACKGROUND,
                radius=18,
                inset=10)
            surface.grid(
                row=row,
                column=0,
                sticky="ew",
                pady=(0, 9))
            direction_surfaces[direction.key] = surface
            card_panel = surface.interior
            card_panel.columnconfigure(0, weight=1)
            card_panel.columnconfigure(2, weight=1)

            enabled = tk.BooleanVar(value=card.enabled)
            enabled_variables[direction.key] = enabled
            ttk.Checkbutton(
                card_panel,
                text=direction.name,
                variable=enabled,
                command=lambda key=language.key: (
                    self._card_controls_changed(key)),
                style="CardOption.TCheckbutton").grid(
                    row=0,
                    column=0,
                    sticky="w")
            ttk.Label(
                card_panel,
                text="DECK",
                style="PaleField.TLabel").grid(
                    row=0,
                    column=1,
                    sticky="e",
                    padx=(14, 7))
            deck_variable = tk.StringVar(
                value=card.target_deck)
            deck_variables[direction.key] = deck_variable
            deck_box = ttk.Combobox(
                card_panel,
                textvariable=deck_variable,
                values=self.app.deck_options,
                width=25,
                state="normal",
                style="App.TCombobox")
            deck_box.grid(
                row=0,
                column=2,
                sticky="ew")
            deck_box.bind(
                "<FocusOut>",
                self.app._clear_entry_selection,
                add="+")
            deck_box.bind(
                "<Button-1>",
                self.app._deck_selector_opened,
                add="+")
            deck_boxes[direction.key] = deck_box

            ttk.Label(
                card_panel,
                text=direction.description,
                style="CardDescription.TLabel",
                wraplength=850).grid(
                    row=(3 if direction.key == "context" else 2),
                    column=0,
                    columnspan=3,
                    sticky="w",
                    pady=(5, 6))
            enhanced = tk.BooleanVar(value=card.enhanced)
            enhanced_variables[direction.key] = enhanced
            backend_label = enhanced_audio_backend_label(language.key)
            enhanced_check = ttk.Checkbutton(
                card_panel,
                text=f"Enhanced · local audio · {backend_label}",
                variable=enhanced,
                command=lambda key=language.key: (
                    self._card_controls_changed(key)),
                state=(
                    tk.DISABLED
                    if language.key == "latin"
                    else tk.NORMAL),
                style="CardOption.TCheckbutton")
            enhanced_check.grid(
                row=1,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(5, 0))
            enhanced_checks[direction.key] = enhanced_check
            if direction.key == "context":
                sentence_translation_check = ttk.Checkbutton(
                    card_panel,
                    text="Include sentence translations",
                    variable=include_sentence_translations,
                    command=lambda key=language.key: (
                        self._card_controls_changed(key)),
                    style="CardOption.TCheckbutton")
                sentence_translation_check.grid(
                    row=2,
                    column=0,
                    columnspan=3,
                    sticky="w",
                    pady=(5, 0))
            (
                field_grid,
                field_enabled,
                field_languages,
                field_boxes,
            ) = self._create_field_grid(
                card_panel,
                language,
                card.fields,
                card.field_languages)
            field_grid.grid(
                row=(4 if direction.key == "context" else 3),
                column=0,
                columnspan=3,
                sticky="ew")
            field_containers[direction.key] = field_grid
            field_enabled_by_direction[
                direction.key] = field_enabled
            field_languages_by_direction[
                direction.key] = field_languages
            field_boxes_by_direction[
                direction.key] = field_boxes
            self._trace(enabled)
            self._trace(enhanced)
            self._trace(deck_variable)

        self.direction_enabled_variables[
            language.key] = enabled_variables
        self.direction_enhanced_variables[
            language.key] = enhanced_variables
        self.direction_enhanced_checks[
            language.key] = enhanced_checks
        self.sentence_translation_variables[
            language.key] = include_sentence_translations
        self.sentence_translation_checks[
            language.key] = sentence_translation_check
        self._trace(include_sentence_translations)
        self.direction_target_deck_variables[
            language.key] = deck_variables
        self.direction_target_deck_boxes[
            language.key] = deck_boxes
        self.direction_surfaces[
            language.key] = direction_surfaces
        self.per_card_field_enabled_variables[
            language.key] = field_enabled_by_direction
        self.per_card_field_language_variables[
            language.key] = field_languages_by_direction
        self.per_card_field_language_boxes[
            language.key] = field_boxes_by_direction
        self.per_card_field_containers[
            language.key] = field_containers

    def _fields_from_controls(
            self,
            enabled_variables,
            language_variables):
        return tuple(
            pipeline_store.FieldSetting(
                field_key=field.key,
                target_language_key=self._language_key_from_name(
                    language_variables[field.key].get()))
            for field in pipeline_store.list_field_options()
            if enabled_variables[field.key].get())

    def _all_field_languages_from_controls(
            self,
            language_variables):
        return tuple(
            pipeline_store.FieldSetting(
                field_key=field.key,
                target_language_key=self._language_key_from_name(
                    language_variables[field.key].get()))
            for field in pipeline_store.list_field_options())

    def _layout_mode_changed(self, language_key):
        self._update_visibility(language_key)
        self.app.schedule_pipeline_save()
        self.app._update_pipeline_status()

    def _learned_filter_changed(self, language_key):
        self.app._set_language_filter_enabled(
            language_key,
            self.learned_filter_enabled_variables[
                language_key].get())
        self._update_learned_filter_summary(language_key)

    def _update_learned_filter_summary(self, language_key):
        settings = self.app._language_filter(language_key)
        self.learned_filter_enabled_variables[
            language_key].set(settings.enabled)
        complete = sum(source.complete for source in settings.sources)
        if not settings.enabled:
            text = "Off for this language"
        elif not complete:
            text = "On · add a deck, note type, and field"
        else:
            text = (
                f"{complete} learned field"
                f"{'' if complete == 1 else 's'} configured")
            if complete != len(settings.sources):
                text += " · one row is incomplete"
        self.learned_filter_summary_variables[
            language_key].set(text)

    def refresh_learned_filter_controls(self):
        for language in pipeline_store.list_settings_languages():
            if language.key not in self.built_language_keys:
                continue
            self._update_learned_filter_summary(language.key)

    def _field_controls_changed(self, language_key):
        self._update_field_box_states(language_key)
        self.app.schedule_pipeline_save()
        self.app._update_pipeline_status()

    def _card_controls_changed(self, language_key):
        self._update_visibility(language_key)
        self.app.schedule_pipeline_save()
        self.app._update_pipeline_status()

    def _update_field_box_states(self, language_key):
        for _field_key, box in (
                self.shared_field_language_boxes[
                    language_key].items()):
            box.configure(state="readonly")
        for _direction_key, boxes in (
                self.per_card_field_language_boxes[
                    language_key].items()):
            for box in boxes.values():
                box.configure(state="readonly")

    def _update_visibility(self, language_key):
        same_deck = self.separate_target_deck_variables[
            language_key].get()
        separate = not same_deck
        sharing = self.share_field_settings_variables[
            language_key].get()
        concrete_language_key = self._concrete_source_language_key(
            language_key)
        enhanced_supported = pipeline_store.enhanced_audio_supported(
            concrete_language_key)
        if separate:
            self.target_deck_boxes[language_key].configure(
                state="disabled")
        else:
            self.target_deck_boxes[language_key].configure(
                state="normal")
        if sharing:
            self.shared_field_containers[language_key].grid()
        else:
            self.shared_field_containers[language_key].grid_remove()
        for direction in pipeline_store.list_directions():
            enabled = self.direction_enabled_variables[
                language_key][direction.key].get()
            enhanced_check = self.direction_enhanced_checks[
                language_key][direction.key]
            enhanced_check.configure(
                text=(
                    "Enhanced · unavailable for "
                    f"{pipeline_store.get_language(concrete_language_key).name}"
                    " · modern-language choice retained"
                    if not enhanced_supported
                    else (
                        "Enhanced · local audio · "
                        + enhanced_audio_backend_label(language_key)
                    )),
                state=(
                    tk.DISABLED
                    if not enhanced_supported or not enabled
                    else tk.NORMAL))
            self.direction_target_deck_boxes[
                language_key][direction.key].configure(
                    state=(
                        "normal"
                        if separate and enabled
                        else "disabled"))
            container = self.per_card_field_containers[
                language_key][direction.key]
            if sharing:
                container.grid_remove()
            else:
                container.grid()
        sentence_translation_checks = getattr(
            self, "sentence_translation_checks", {})
        if language_key in sentence_translation_checks:
            sentence_translation_checks[language_key].configure(
                state=(
                    tk.NORMAL
                    if self.direction_enabled_variables[
                        language_key]["context"].get()
                    else tk.DISABLED))
        self._update_field_box_states(language_key)
        self._refresh_layout_geometry(language_key)

    def _refresh_layout_geometry(self, language_key):
        """Let Tk's configure events resize surfaces without a forced flush.

        Every ``RoundedPanel`` already tracks its interior's ``<Configure>``
        event. Forcing several synchronous ``update_idletasks`` passes here
        made startup lay out all five language editors repeatedly before the
        window could be shown.
        """
        if not self.frame.winfo_exists():
            return
        scroll_frame = getattr(
            self.app,
            "pipeline_scroll_frame",
            None)
        if scroll_frame is not None:
            scroll_frame._content_changed()

    def _update_all_visibility(self):
        for language in pipeline_store.list_settings_languages():
            if language.key not in self.built_language_keys:
                continue
            self._update_visibility(language.key)

    def refresh_generation_language(self):
        """Refresh translation targets after the Generate variant changes."""
        for language in pipeline_store.list_settings_languages():
            settings_key = language.key
            if settings_key not in self.built_language_keys:
                continue
            concrete_source_key = self._concrete_source_language_key(
                settings_key)
            groups = [
                (
                    self.shared_field_language_variables[settings_key],
                    self.shared_field_language_boxes[settings_key],
                ),
            ]
            groups.extend(
                (
                    self.per_card_field_language_variables[
                        settings_key][direction.key],
                    self.per_card_field_language_boxes[
                        settings_key][direction.key],
                )
                for direction in pipeline_store.list_directions())
            for variables, boxes in groups:
                for field in pipeline_store.list_field_options():
                    values = self._language_values(
                        concrete_source_key,
                        field.key)
                    boxes[field.key].configure(values=values)
                    if field.key != "translation":
                        continue
                    selected_name = variables[field.key].get()
                    try:
                        selected_key = self._language_key_from_name(
                            selected_name)
                    except ValueError:
                        selected_key = None
                    if (
                            selected_key is not None
                            and pipeline_store.translation_target_allowed(
                                concrete_source_key,
                                selected_key)):
                        continue
                    replacement_key = self._default_field_language_key(
                        concrete_source_key,
                        field.key)
                    variables[field.key].set(
                        pipeline_store.get_language(
                            replacement_key).name)
            self._update_visibility(settings_key)

    def _language_changed(self, _event=None):
        language = self.get_active_language()
        newly_built = self._ensure_language_tab_built(language.key)
        if (
                language.key == self.last_active_language_key
                and not newly_built):
            return
        self.last_active_language_key = language.key
        self._update_visibility(language.key)
        if newly_built:
            scroll_frame = getattr(
                self.app,
                "pipeline_scroll_frame",
                None)
            if scroll_frame is not None:
                scroll_frame.bind_mousewheel_tree()
            self.bind_target_deck_mousewheel()

    def get_active_language(self):
        selected_tab = self.language_notebook.select()
        language = self.language_by_tab.get(selected_tab)
        if language is None:
            raise ValueError("Select a language.")
        return language

    def _settings_from_built_language(self, language):
        """Read one materialized language page without touching lazy pages."""
        shared_fields = self._fields_from_controls(
            self.shared_field_enabled_variables[language.key],
            self.shared_field_language_variables[language.key])
        shared_field_languages = (
            self._all_field_languages_from_controls(
                self.shared_field_language_variables[
                    language.key]))
        cards = []
        for direction in pipeline_store.list_directions():
            fields = self._fields_from_controls(
                self.per_card_field_enabled_variables[
                    language.key][direction.key],
                self.per_card_field_language_variables[
                    language.key][direction.key])
            cards.append(pipeline_store.CardSettings(
                direction_key=direction.key,
                enabled=self.direction_enabled_variables[
                    language.key][direction.key].get(),
                fields=fields,
                target_deck=self.direction_target_deck_variables[
                    language.key][direction.key].get().strip(),
                field_languages=(
                    self._all_field_languages_from_controls(
                        self.per_card_field_language_variables[
                            language.key][direction.key])),
                enhanced=self.direction_enhanced_variables[
                    language.key][direction.key].get()))
        return pipeline_store.LanguageSettings(
            language_key=language.key,
            cards=tuple(cards),
            target_deck=self.target_deck_variables[
                language.key].get().strip(),
            separate_target_decks=(
                not self.separate_target_deck_variables[
                    language.key].get()),
            share_field_settings=(
                self.share_field_settings_variables[
                    language.key].get()),
            make_items_for_characters=(
                self.make_items_for_characters_variables[
                    language.key].get()
                if language.key in self.make_items_for_characters_variables
                else False),
            include_sentence_translations=(
                self.sentence_translation_variables[language.key].get()),
            shared_fields=shared_fields,
            shared_field_languages=shared_field_languages)

    def to_config(self):
        active_language = self.app.get_card_setup_generation_language()
        all_settings = []
        for language in pipeline_store.list_settings_languages():
            if language.key not in self.built_language_keys:
                all_settings.append(
                    pipeline_store.get_language_settings(
                        self.config,
                        language.key))
                continue
            all_settings.append(
                self._settings_from_built_language(language))
        active_settings = next(
            settings
            for settings in all_settings
            if settings.language_key
            == active_language.model_language_key)
        return pipeline_store.replace_active_language_settings(
            self.config,
            active_settings,
            all_settings,
            active_language_key=active_language.key)

    def update_deck_options(self, deck_options):
        for box in self.target_deck_boxes.values():
            box.configure(values=deck_options)
        for boxes in self.direction_target_deck_boxes.values():
            for box in boxes.values():
                box.configure(values=deck_options)

    def bind_target_deck_mousewheel(self):
        """Scroll Card Setup without changing a deck under the pointer."""
        scroll_frame = self.app.pipeline_scroll_frame

        def scroll_page(event):
            amount = wheel_scroll_amount(event)
            if amount:
                scroll_frame.canvas.yview_scroll(amount, "units")
            # Stop ttk::combobox's class binding from selecting the previous
            # or next deck after the page-scroll handler has run.
            return "break"

        for box in (
                *self.target_deck_boxes.values(),
                *(
                    box
                    for boxes in self.direction_target_deck_boxes.values()
                    for box in boxes.values())):
            box.bind("<MouseWheel>", scroll_page)
            box.bind("<Button-4>", scroll_page)
            box.bind("<Button-5>", scroll_page)

    def destroy(self):
        for variable, trace_id in self.variable_traces:
            variable.trace_remove("write", trace_id)
        self.frame.destroy()


class AutoAnkiApp:
    POLL_INTERVAL_MS = 100
    DECK_REFRESH_INTERVAL_MS = 5000
    WINDOW_BACKGROUND = "#F2F1ED"
    PANEL_BACKGROUND = "#FFFFFF"
    TEXT_PRIMARY = "#182629"
    TEXT_SECONDARY = "#667376"
    LINE = "#D9DDDA"
    ACCENT = "#176C64"
    ACCENT_HOVER = "#207B72"
    ACCENT_PRESSED = "#0E4C46"
    PALE = "#E7F1EE"
    DARK = "#172D31"
    WARNING = "#B06A21"
    ERROR = "#B94B45"

    def __init__(
            self,
            root,
            pipeline_executor=None,
            pipeline_loader=None,
            pipeline_saver=None,
            source_catalog_loader=None,
            source_estimator=None,
            source_generate_callback=None,
            source_prepare_callback=None,
            source_codex_callback=None,
            source_jobs_loader=None,
            source_pause_callback=None,
            source_resume_callback=None,
            source_dispatch_status_loader=None,
            paid_dispatch_control=None,
            source_retry_callback=None,
            source_inspect_callback=None,
            source_validation_accept_callback=None,
            source_anki_options_loader=None,
            source_preview_loader=None,
            manual_input_filter_callback=None,
            learned_filter_loader=None,
            learned_filter_saver=None,
            preference_loader=None,
            preference_saver=None):
        self.root = root
        try:
            native_scaling = float(
                self.root.tk.call("tk", "scaling"))
            if native_scaling > 1.35:
                self.root.tk.call("tk", "scaling", 1.35)
        except (AttributeError, tk.TclError, TypeError, ValueError):
            pass
        self.pipeline_executor = (
            pipeline_executor or pipeline_runner.run_pipelines)
        self.pipeline_loader = (
            pipeline_loader or pipeline_store.load_pipelines)
        self.pipeline_saver = (
            pipeline_saver or pipeline_store.save_pipelines)
        # Source-generation hooks are deliberately injected. Merely opening
        # the UI can only perform catalogue/job reads and offline estimation;
        # paid generation, PDF processing, and Codex retrieval require an
        # explicit button press.
        self.source_catalog_loader = source_catalog_loader
        self.source_estimator = source_estimator
        self.source_generate_callback = source_generate_callback
        self.source_prepare_callback = source_prepare_callback
        self.source_codex_callback = source_codex_callback
        self.source_jobs_loader = source_jobs_loader
        self.source_pause_callback = source_pause_callback
        self.source_resume_callback = source_resume_callback
        self.source_dispatch_status_loader = (
            source_dispatch_status_loader)
        self.paid_dispatch_control = paid_dispatch_control
        self.source_retry_callback = source_retry_callback
        self.source_inspect_callback = source_inspect_callback
        self.source_validation_accept_callback = (
            source_validation_accept_callback)
        self.source_anki_options_loader = source_anki_options_loader
        self.source_preview_loader = source_preview_loader
        self.manual_input_filter_callback = manual_input_filter_callback
        self.learned_filter_loader = (
            learned_filter_loader
            or learned_filter_store.load_language_filters)
        self.learned_filter_saver = (
            learned_filter_saver
            or learned_filter_store.save_language_filters)
        self.preference_loader = (
            preference_loader or gui_preferences.load_preferences)
        self.preference_saver = (
            preference_saver or gui_preferences.save_preferences)
        try:
            loaded_preferences = self.preference_loader()
            self.gui_preferences = gui_preferences.default_preferences()
            self.gui_preferences.update({
                key: value
                for key, value in dict(loaded_preferences).items()
                if key in self.gui_preferences
            })
            self.preference_load_error = None
        except (OSError, TypeError, ValueError) as error:
            self.gui_preferences = gui_preferences.default_preferences()
            self.preference_load_error = error
        self._preferences_loading = True
        self.preference_save_after_id = None
        self.preference_variable_traces = []
        self.result_queue = queue.Queue()
        self.deck_result_queue = queue.Queue()
        self.connection_result_queue = queue.Queue()
        self.source_action_result_queue = queue.Queue()
        self.source_anki_options_queue = queue.Queue()
        self.source_preview_result_queue = queue.Queue()
        self.pipeline_rows = []
        self.pipeline_save_after_id = None
        self.pipeline_notice_after_id = None
        self.deck_refresh_in_progress = False
        self.deck_launch_requested = False
        self.generation_in_progress = False
        self.source_action_in_progress = False
        self.source_estimate_after_id = None
        self.source_anki_options_pending = 0
        self.source_anki_options_polling = False
        self.source_preview_generation = 0
        self.source_preview_pending = set()
        self.source_preview_polling = False
        self._pipeline_tab_built = False
        self._from_source_tab_built = False
        self._advanced_tab_built = False
        self._help_tab_built = False
        self._tracked_preference_variable_ids = set()
        self._source_preference_tracking_installed = False

        root.title("AutoAnki")
        root.configure(background=self.WINDOW_BACKGROUND)

        self.status = tk.StringVar(value="Ready to generate")
        self.output = tk.StringVar(
            value=str(process_text.DECK_PATH))
        self.input_count = tk.StringVar(
            value="0 lines · 0 characters")
        self.pipeline_status = tk.StringVar(
            value="Saved card settings load when this tab is opened.")
        self.deck_status = tk.StringVar(
            value="Checking Anki deck list…")
        try:
            anki_settings = (
                anki_integration.load_anki_connection_settings())
            connection_status = (
                "Saved locally · test the connection after changing it")
        except (OSError, ValueError) as error:
            anki_settings = anki_integration.AnkiConnectionSettings()
            connection_status = f"Could not read saved settings · {error}"
        self.anki_url = tk.StringVar(value=anki_settings.url)
        self.anki_api_key = tk.StringVar(
            value=anki_settings.api_key or "")
        self.anki_connection_status = tk.StringVar(
            value=connection_status)
        # These variables mirror the active Manual and From Source language.
        # The actual retained configuration is language-specific and may
        # contain any number of deck/note-type/field combinations.
        self.source_exclude_enabled = tk.BooleanVar(value=False)
        self.manual_filter_enabled = tk.BooleanVar(value=False)
        self.source_exclusion_status = tk.StringVar(
            value="Learned-word filter is off.")
        self.manual_filter_summary = tk.StringVar(
            value="Learned-word filter is off")
        try:
            self.learned_filter_settings = tuple(
                self.learned_filter_loader())
            self.learned_filter_load_error = None
        except (OSError, TypeError, ValueError) as error:
            self.learned_filter_settings = ()
            self.learned_filter_load_error = error
        self.learned_filter_dialog = None
        self.learned_filter_dialog_language_key = None
        self.learned_filter_rows = []
        self.learned_filter_rows_by_id = {}

        try:
            loaded_pipelines = tuple(self.pipeline_loader())
            self.pipeline_configs = (
                loaded_pipelines[:1]
                or (pipeline_store.default_pipeline(),))
            self.pipeline_load_error = None
        except (OSError, ValueError) as error:
            self.pipeline_configs = (
                pipeline_store.default_pipeline(),)
            self.pipeline_load_error = error
        initial_language = pipeline_store.get_language(
            self.pipeline_configs[0].language_key)
        self.generation_language = tk.StringVar(
            value=initial_language.name)

        configured_decks = set()
        for pipeline in self.pipeline_configs:
            settings_items = (
                pipeline_store.get_language_settings(
                    pipeline,
                    pipeline.language_key),
                *pipeline.language_settings,
            )
            for settings in settings_items:
                if settings.target_deck:
                    configured_decks.add(settings.target_deck)
                configured_decks.update(
                    card.target_deck
                    for card in settings.cards
                    if card.target_deck)
        try:
            cached_decks = pipeline_store.load_anki_deck_cache()
        except (OSError, ValueError) as error:
            cached_decks = ()
            self.deck_status.set(
                f"Could not read saved Anki decks · {error}")
        else:
            if cached_decks:
                self.deck_status.set(
                    "Showing saved decks while checking Anki…")
        self.deck_options = tuple(sorted(
            configured_decks | set(cached_decks)))

        self._configure_styles()
        self._install_combobox_wheel_guard()
        self._build_widgets()
        self._set_initial_window_size()
        self._preferences_loading = False
        self._install_preference_tracking()
        self.root.bind(
            "<Button-1>",
            self._clear_focus_on_background_click,
            add="+")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.notebook.bind(
            "<<NotebookTabChanged>>",
            self._main_tab_changed,
            add="+")
        self.root.after(0, self.refresh_anki_decks)
        self.root.after_idle(self._restore_navigation_preferences)

        if self.pipeline_load_error:
            root.after(0, self._show_pipeline_load_error)
        if self.preference_load_error:
            self.pipeline_status.set(
                "Could not read saved GUI preferences; safe defaults were "
                f"loaded · {self.preference_load_error}")

    def _show_pipeline_load_error(self):
        messagebox.showerror(
            "Could not load pipeline settings",
            (
                f"{self.pipeline_load_error}\n\n"
                "The default pipeline was loaded instead."),
            parent=self.root)

    @staticmethod
    def _selected_tab_key(notebook, tabs, fallback):
        selected = notebook.select()
        for key, tab in tabs.items():
            if selected == str(tab):
                return key
        return fallback

    def _collect_gui_preferences(self):
        preferences = gui_preferences.default_preferences()
        preferences.update({
            key: value
            for key, value in self.gui_preferences.items()
            if key in preferences
        })
        if hasattr(self, "prompt_selector_value"):
            preferences["prompt_key"] = self.prompt_selector_value.get()
        if hasattr(self, "input_text"):
            preferences["manual_input_draft"] = self.input_text.get(
                "1.0",
                "end-1c")

        if self._from_source_tab_built:
            option = self.source_options_by_label.get(
                self.source_selected_label.get())
            preferences["source_key"] = (
                option.key if option is not None else "")
            language_overrides = dict(
                preferences.get("source_language_overrides", {}))
            if option is not None:
                selected_language = next((
                    language
                    for language in pipeline_store.list_languages()
                    if language.name == self.source_language_label.get()
                ), None)
                if selected_language is not None:
                    language_overrides[option.key] = selected_language.key
            preferences["source_language_overrides"] = language_overrides

            for key, variable_name in (
                    ("source_chunk_size", "source_chunk_size"),
                    (
                        "source_prefix_token_limit",
                        "source_prefix_token_limit",
                    ),
                    ("source_concurrency", "source_concurrency"),
                    (
                        "source_request_stagger_ms",
                        "source_request_stagger_ms",
                    ),
                    (
                        "source_preview_page_size",
                        "source_preview_page_size",
                    ),
                    ("source_file_language", "source_file_language"),
                    ("source_codex_language", "source_codex_language")):
                preferences[key] = getattr(self, variable_name).get()
            for key, variable_name in (
                    ("source_limit_to_prefix", "source_limit_to_prefix"),
                    (
                        "source_allow_web_search",
                        "source_allow_web_search",
                    ),
                    (
                        "source_use_source_examples",
                        "source_use_source_examples",
                    ),
                    (
                        "source_include_context_nuance",
                        "source_include_context_nuance",
                    ),
                    ("source_separate_decks", "source_separate_decks"),
                    (
                        "source_automatic_repair",
                        "source_automatic_repair",
                    ),
                    ("source_file_use_gpu", "source_file_use_gpu")):
                preferences[key] = bool(
                    getattr(self, variable_name).get())

            preferences["source_card_directions"] = list(
                self._selected_source_direction_keys())
            preferences["source_model_key"] = (
                self._selected_source_model_key())
            preferences["source_protocol_key"] = (
                SOURCE_PROTOCOL_KEYS_BY_LABEL.get(
                    self.source_request_protocol_label.get(),
                    "v10"))
            current_protocol = preferences["source_protocol_key"]
            preferences["source_reasoning_key"] = (
                getattr(self, "_source_reasoning_before_v8", "low")
                if current_protocol == "v8"
                else SOURCE_REASONING_KEYS_BY_LABEL.get(
                    self.source_reasoning_label.get(),
                    "low"))
            preferences["source_execution_key"] = (
                SOURCE_EXECUTION_KEYS_BY_LABEL.get(
                    self.source_execution_label.get(),
                    "standard"))
            preferences["source_context_key"] = (
                SOURCE_CONTEXT_KEYS_BY_LABEL.get(
                    self.source_context_label.get(),
                    "sentence"))
            preferences["source_non_example_context_key"] = getattr(
                self,
                "_source_context_before_source_examples",
                "sentence")
            preferences["source_tab"] = self._selected_tab_key(
                self.source_notebook,
                {
                    "generate": self.source_generate_page,
                    "inspect": self.source_preview_page,
                    "prepare": self.source_prepare_page,
                    "codex": self.source_codex_page,
                    "jobs": self.source_jobs_page,
                },
                "generate")

        preferences["main_tab"] = self._selected_tab_key(
            self.notebook,
            {
                "generate": self.generate_tab,
                "card_setup": self.pipeline_tab,
                "advanced": self.advanced_tab,
                "help": self.help_tab,
            },
            "generate")
        preferences["generate_tab"] = self._selected_tab_key(
            self.generate_notebook,
            {
                "manual": self.manual_generate_tab,
                "source": self.from_source_tab,
            },
            "manual")
        return preferences

    def _track_preference_variables(self, variables):
        for variable in variables:
            variable_id = id(variable)
            if variable_id in self._tracked_preference_variable_ids:
                continue
            trace_id = variable.trace_add(
                "write",
                self._schedule_preferences_save)
            self.preference_variable_traces.append((variable, trace_id))
            self._tracked_preference_variable_ids.add(variable_id)

    def _install_preference_tracking(self):
        if self._advanced_tab_built:
            self._install_advanced_preference_tracking()
        for notebook in (
                self.notebook,
                self.generate_notebook):
            notebook.bind(
                "<<NotebookTabChanged>>",
                self._schedule_preferences_save,
                add="+")
        if self._from_source_tab_built:
            self._install_source_preference_tracking()

    def _install_advanced_preference_tracking(self):
        if hasattr(self, "prompt_selector_value"):
            self._track_preference_variables([
                self.prompt_selector_value,
            ])

    def _install_source_preference_tracking(self):
        if self._source_preference_tracking_installed:
            return
        self._source_preference_tracking_installed = True
        self._track_preference_variables([
            self.source_selected_label,
            self.source_language_label,
            self.source_chunk_size,
            self.source_limit_to_prefix,
            self.source_prefix_token_limit,
            self.source_concurrency,
            self.source_request_stagger_ms,
            self.source_allow_web_search,
            self.source_use_source_examples,
            self.source_include_context_nuance,
            self.source_separate_decks,
            self.source_model_label,
            self.source_request_protocol_label,
            self.source_reasoning_label,
            self.source_execution_label,
            self.source_automatic_repair,
            self.source_context_label,
            self.source_preview_page_size,
            self.source_file_language,
            self.source_file_use_gpu,
            self.source_codex_language,
            *self.source_card_direction_variables.values(),
        ])
        self.source_notebook.bind(
            "<<NotebookTabChanged>>",
            self._schedule_preferences_save,
            add="+")

    def _schedule_preferences_save(self, *_args):
        if self._preferences_loading:
            return
        if self.preference_save_after_id is not None:
            self.root.after_cancel(self.preference_save_after_id)
        self.preference_save_after_id = self.root.after(
            350,
            self._save_gui_preferences)

    def _save_gui_preferences(self):
        self.preference_save_after_id = None
        try:
            preferences = self._collect_gui_preferences()
            saved = self.preference_saver(preferences)
        except (OSError, TypeError, ValueError) as error:
            if hasattr(self, "source_action_status"):
                self.source_action_status.set(
                    f"Could not save GUI preferences · {error}")
            return False
        self.gui_preferences = (
            dict(saved) if isinstance(saved, dict) else preferences)
        return True

    def _restore_navigation_preferences(self):
        self._preferences_loading = True
        try:
            generate_tabs = {
                "manual": self.manual_generate_tab,
                "source": self.from_source_tab,
            }
            generate_tab = generate_tabs.get(
                self.gui_preferences.get("generate_tab"))
            if generate_tab is not None:
                if generate_tab == self.from_source_tab:
                    self._ensure_from_source_tab_built()
                self.generate_notebook.select(generate_tab)

            main_tabs = {
                "generate": self.generate_tab,
                "card_setup": self.pipeline_tab,
                "advanced": self.advanced_tab,
                "help": self.help_tab,
            }
            main_tab = main_tabs.get(
                self.gui_preferences.get("main_tab"))
            if main_tab is not None:
                if main_tab == self.pipeline_tab:
                    self._ensure_pipeline_tab_built()
                elif main_tab == self.advanced_tab:
                    self._ensure_advanced_tab_built()
                elif main_tab == self.help_tab:
                    self._ensure_help_tab_built()
                self.notebook.select(main_tab)
        finally:
            self._preferences_loading = False

    def _set_initial_window_size(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = min(1860, screen_width - 80)
        height = min(1200, screen_height - 100)
        x_position = max((screen_width - width) // 2, 0)
        y_position = max((screen_height - height) // 2, 0)

        self.root.geometry(
            f"{width}x{height}+{x_position}+{y_position}")
        self.root.minsize(min(1000, width), min(720, height))

    def _install_combobox_wheel_guard(self):
        """Never let a closed dropdown change merely because it was hovered."""
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_class(
                "TCombobox",
                sequence,
                lambda _event: "break")

    def _configure_styles(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self._style_images = []
        rounded_mask_cache = {}
        checkbox_mask_cache = {}

        def rounded_tile(
                size,
                radius,
                fill,
                outline=None,
                backdrop=None):
            image = tk.PhotoImage(
                master=self.root,
                width=size,
                height=size)
            backdrop = backdrop or fill

            def inside(x, y, inset=0):
                low = inset
                high = size - inset
                curve = max(0.5, radius - inset)
                if not (low <= x <= high and low <= y <= high):
                    return False
                nearest_x = min(
                    max(x, low + curve),
                    high - curve)
                nearest_y = min(
                    max(y, low + curve),
                    high - curve)
                return (
                    (x - nearest_x) ** 2
                    + (y - nearest_y) ** 2
                    <= curve ** 2)

            def rgb(colour):
                value = colour.lstrip("#")
                return tuple(
                    int(value[index:index + 2], 16)
                    for index in (0, 2, 4))

            samples = 4
            sample_count = samples * samples
            mask_key = (size, radius)
            masks = rounded_mask_cache.get(mask_key)
            if masks is None:
                masks = []
                for y in range(size):
                    mask_row = []
                    for x in range(size):
                        fill_count = 0
                        outline_count = 0
                        backdrop_count = 0
                        for sample_y in range(samples):
                            point_y = y + (sample_y + 0.5) / samples
                            for sample_x in range(samples):
                                point_x = x + (
                                    sample_x + 0.5) / samples
                                if inside(point_x, point_y, 1):
                                    fill_count += 1
                                elif inside(point_x, point_y):
                                    outline_count += 1
                                else:
                                    backdrop_count += 1
                        mask_row.append((
                            fill_count,
                            outline_count,
                            backdrop_count))
                    masks.append(tuple(mask_row))
                masks = tuple(masks)
                rounded_mask_cache[mask_key] = masks

            fill_rgb = rgb(fill)
            outline_rgb = rgb(outline or fill)
            backdrop_rgb = rgb(backdrop)
            image_rows = []
            for mask_row in masks:
                image_row = []
                for fill_count, outline_count, backdrop_count in mask_row:
                    colour = tuple(
                        round((
                            fill_rgb[channel] * fill_count
                            + outline_rgb[channel] * outline_count
                            + backdrop_rgb[channel] * backdrop_count
                        ) / sample_count)
                        for channel in range(3))
                    image_row.append("#%02x%02x%02x" % colour)
                image_rows.append(image_row)
            image.put(" ".join(
                "{" + " ".join(row) + "}"
                for row in image_rows))
            self._style_images.append(image)
            return image

        def checkbox_image(
                *,
                fill,
                outline,
                selected=False,
                check="#FFFFFF",
                backdrop=None):
            width = 29
            height = 22
            box_size = 22
            radius = 7.0
            samples = 4
            sample_count = samples * samples
            image = tk.PhotoImage(
                master=self.root,
                width=width,
                height=height)

            def colour_rgb(colour):
                value = colour.lstrip("#")
                return tuple(
                    int(value[index:index + 2], 16)
                    for index in (0, 2, 4))

            def inside(x, y, inset=0):
                low = inset
                high = box_size - inset
                curve = max(0.5, radius - inset)
                if not (low <= x <= high and low <= y <= high):
                    return False
                nearest_x = min(
                    max(x, low + curve),
                    high - curve)
                nearest_y = min(
                    max(y, low + curve),
                    high - curve)
                return (
                    (x - nearest_x) ** 2
                    + (y - nearest_y) ** 2
                    <= curve ** 2)

            def segment_distance(x, y, start, end):
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length_squared = dx * dx + dy * dy
                projection = (
                    (x - start[0]) * dx
                    + (y - start[1]) * dy
                ) / length_squared
                projection = max(0.0, min(1.0, projection))
                nearest_x = start[0] + projection * dx
                nearest_y = start[1] + projection * dy
                return (
                    (x - nearest_x) ** 2
                    + (y - nearest_y) ** 2
                ) ** 0.5

            check_segments = (
                ((5.4, 11.2), (9.2, 14.8)),
                ((9.2, 14.8), (16.8, 7.2)))
            masks = checkbox_mask_cache.get(selected)
            if masks is None:
                masks = []
                for y in range(height):
                    mask_row = []
                    for x in range(width):
                        counts = [0, 0, 0, 0]
                        for sample_y in range(samples):
                            point_y = y + (sample_y + 0.5) / samples
                            for sample_x in range(samples):
                                point_x = x + (
                                    sample_x + 0.5) / samples
                                on_check = selected and any(
                                    segment_distance(
                                        point_x,
                                        point_y,
                                        start,
                                        end) <= 1.35
                                    for start, end in check_segments)
                                if on_check:
                                    counts[0] += 1
                                elif inside(point_x, point_y, 1):
                                    counts[1] += 1
                                elif inside(point_x, point_y):
                                    counts[2] += 1
                                else:
                                    counts[3] += 1
                        mask_row.append(tuple(counts))
                    masks.append(tuple(mask_row))
                masks = tuple(masks)
                checkbox_mask_cache[selected] = masks

            colours = (
                colour_rgb(check),
                colour_rgb(fill),
                colour_rgb(outline),
                colour_rgb(backdrop or self.PALE),
            )
            image_rows = []
            for mask_row in masks:
                image_row = []
                for counts in mask_row:
                    colour = tuple(
                        round(sum(
                            colours[kind][channel] * count
                            for kind, count in enumerate(counts)
                        ) / sample_count)
                        for channel in range(3))
                    image_row.append("#%02x%02x%02x" % colour)
                image_rows.append(image_row)
            image.put(" ".join(
                "{" + " ".join(row) + "}"
                for row in image_rows))
            self._style_images.append(image)
            return image

        def rounded_button_style(
                style_name,
                element_name,
                *,
                normal,
                active,
                pressed,
                disabled,
                outline,
                backdrop):
            normal_image = rounded_tile(
                44, 20, normal, outline, backdrop)
            active_image = rounded_tile(
                44, 20, active, active, backdrop)
            pressed_image = rounded_tile(
                44, 20, pressed, pressed, backdrop)
            disabled_image = rounded_tile(
                44, 20, disabled, disabled, backdrop)
            style.element_create(
                element_name,
                "image",
                normal_image,
                ("disabled", disabled_image),
                ("pressed", pressed_image),
                ("active", active_image),
                border=(20, 20, 20, 20),
                sticky="nsew")
            style.layout(
                style_name,
                [(
                    element_name,
                    {
                        "sticky": "nsew",
                        "children": [(
                            "Button.padding",
                            {
                                "sticky": "nsew",
                                "children": [(
                                    "Button.label",
                                    {"sticky": "nsew"})],
                            })],
                    })])

        style.configure(
            "App.TFrame",
            background=self.WINDOW_BACKGROUND)
        style.configure(
            "Panel.TFrame",
            background=self.PANEL_BACKGROUND)
        style.configure(
            "Pale.TFrame",
            background=self.PALE)
        style.configure(
            "PaleField.TLabel",
            background=self.PALE,
            foreground=self.TEXT_SECONDARY,
            font=("DejaVu Sans", 8, "bold"))
        style.configure(
            "Panel.TCheckbutton",
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            font=("DejaVu Sans", 9, "bold"),
            padding=(0, 3))
        style.map(
            "Panel.TCheckbutton",
            background=[("active", self.PANEL_BACKGROUND)])
        style.configure(
            "Title.TLabel",
            background=self.DARK,
            foreground="#ffffff",
            font=("DejaVu Sans", 25, "bold"))
        style.configure(
            "Subtitle.TLabel",
            background=self.DARK,
            foreground="#A8D3CC",
            font=("DejaVu Sans", 9, "bold"))
        style.configure(
            "Section.TLabel",
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            font=("DejaVu Sans", 11, "bold"))
        style.configure(
            "HelpSection.TLabel",
            background=self.WINDOW_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            font=("DejaVu Sans", 11, "bold"))
        style.configure(
            "Muted.TLabel",
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_SECONDARY,
            font=("DejaVu Sans", 9))
        style.configure(
            "ValidationError.TLabel",
            background=self.PANEL_BACKGROUND,
            foreground=self.ERROR,
            font=("DejaVu Sans", 9, "bold"))
        style.configure(
            "ValidationSafe.TLabel",
            background=self.PANEL_BACKGROUND,
            foreground=self.ACCENT,
            font=("DejaVu Sans", 9, "bold"))
        style.configure(
            "ValidationDialogError.TLabel",
            background=self.WINDOW_BACKGROUND,
            foreground=self.ERROR,
            font=("DejaVu Sans", 9, "bold"))
        style.configure(
            "ValidationDialogSafe.TLabel",
            background=self.WINDOW_BACKGROUND,
            foreground=self.ACCENT,
            font=("DejaVu Sans", 9, "bold"))
        style.configure(
            "Status.TLabel",
            background=self.WINDOW_BACKGROUND,
            foreground=self.TEXT_SECONDARY,
            font=("DejaVu Sans", 9))
        style.configure(
            "FieldLabel.TLabel",
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_SECONDARY,
            font=("DejaVu Sans", 8, "bold"))
        style.configure(
            "CardDescription.TLabel",
            background=self.PALE,
            foreground=self.TEXT_SECONDARY,
            font=("DejaVu Sans", 9))
        style.configure(
            "CardOption.TCheckbutton",
            background=self.PALE,
            foreground=self.TEXT_PRIMARY,
            font=("DejaVu Sans", 10, "bold"),
            padding=(0, 3))
        style.map(
            "CardOption.TCheckbutton",
            background=[
                ("active", self.PALE),
                ("selected", self.PALE)],
            foreground=[
                ("disabled", self.TEXT_SECONDARY),
                ("active", self.TEXT_PRIMARY)])
        check_off = checkbox_image(
            fill=self.PALE,
            outline="#AAB7B3")
        check_off_active = checkbox_image(
            fill="#EEF6F3",
            outline=self.ACCENT)
        check_off_disabled = checkbox_image(
            fill="#F5F6F5",
            outline="#D5DBD8")
        check_on = checkbox_image(
            fill=self.ACCENT,
            outline=self.ACCENT,
            selected=True)
        check_on_active = checkbox_image(
            fill=self.ACCENT_HOVER,
            outline=self.ACCENT_HOVER,
            selected=True)
        check_on_disabled = checkbox_image(
            fill="#AFCAC5",
            outline="#AFCAC5",
            selected=True)
        style.element_create(
            "AutoAnkiModernCheck.indicator",
            "image",
            check_off,
            ("disabled selected", check_on_disabled),
            ("disabled", check_off_disabled),
            ("active selected", check_on_active),
            ("pressed selected", check_on_active),
            ("selected", check_on),
            ("active", check_off_active),
            sticky="")
        style.layout(
            "CardOption.TCheckbutton",
            [(
                "Checkbutton.padding",
                {
                    "sticky": "nswe",
                    "children": [
                        (
                            "AutoAnkiModernCheck.indicator",
                            {"side": "left", "sticky": ""}),
                        (
                            "Checkbutton.label",
                            {"side": "left", "sticky": "w"}),
                    ],
                })])
        panel_check_off = checkbox_image(
            fill=self.PANEL_BACKGROUND,
            outline="#AAB7B3",
            backdrop=self.PANEL_BACKGROUND)
        panel_check_off_active = checkbox_image(
            fill="#EEF6F3",
            outline=self.ACCENT,
            backdrop=self.PANEL_BACKGROUND)
        panel_check_off_disabled = checkbox_image(
            fill="#F5F6F5",
            outline="#D5DBD8",
            backdrop=self.PANEL_BACKGROUND)
        panel_check_on = checkbox_image(
            fill=self.ACCENT,
            outline=self.ACCENT,
            selected=True,
            backdrop=self.PANEL_BACKGROUND)
        panel_check_on_active = checkbox_image(
            fill=self.ACCENT_HOVER,
            outline=self.ACCENT_HOVER,
            selected=True,
            backdrop=self.PANEL_BACKGROUND)
        panel_check_on_disabled = checkbox_image(
            fill="#AFCAC5",
            outline="#AFCAC5",
            selected=True,
            backdrop=self.PANEL_BACKGROUND)
        style.element_create(
            "AutoAnkiPanelModernCheck.indicator",
            "image",
            panel_check_off,
            ("disabled selected", panel_check_on_disabled),
            ("disabled", panel_check_off_disabled),
            ("active selected", panel_check_on_active),
            ("pressed selected", panel_check_on_active),
            ("selected", panel_check_on),
            ("active", panel_check_off_active),
            sticky="")
        style.layout(
            "Panel.TCheckbutton",
            [(
                "Checkbutton.padding",
                {
                    "sticky": "nswe",
                    "children": [
                        (
                            "AutoAnkiPanelModernCheck.indicator",
                            {"side": "left", "sticky": ""}),
                        (
                            "Checkbutton.label",
                            {"side": "left", "sticky": "w"}),
                    ],
                })])
        style.configure(
            "Accent.TButton",
            background=self.ACCENT,
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            padding=(18, 10),
            font=("DejaVu Sans", 10, "bold"))
        style.map(
            "Accent.TButton",
            foreground=[("disabled", "#F3F7F6")])
        style.configure(
            "Secondary.TButton",
            background="#EEF3F1",
            foreground=self.ACCENT,
            bordercolor="#EEF3F1",
            padding=(12, 8),
            font=("DejaVu Sans", 9, "bold"))
        style.configure(
            "CompactSecondary.TButton",
            background="#EEF3F1",
            foreground=self.ACCENT,
            bordercolor="#C8D8D4",
            lightcolor="#EEF3F1",
            darkcolor="#EEF3F1",
            padding=(9, 4),
            font=("DejaVu Sans", 9, "bold"))
        style.map(
            "CompactSecondary.TButton",
            background=[
                ("active", "#DDEBE7"),
                ("pressed", "#D2E3DE"),
                ("disabled", "#F1F3F2")],
            foreground=[("disabled", self.TEXT_SECONDARY)])
        rounded_button_style(
            "Accent.TButton",
            "AutoAnkiPrimaryRounded.button",
            normal=self.ACCENT,
            active=self.ACCENT_HOVER,
            pressed=self.ACCENT_PRESSED,
            disabled="#A9C8C3",
            outline=self.ACCENT,
            backdrop=self.WINDOW_BACKGROUND)
        rounded_button_style(
            "Secondary.TButton",
            "AutoAnkiSecondaryRounded.button",
            normal="#EEF3F1",
            active="#DDEBE7",
            pressed="#D2E3DE",
            disabled="#F1F3F2",
            outline="#EEF3F1",
            backdrop=self.WINDOW_BACKGROUND)
        style.configure(
            "Generation.Horizontal.TProgressbar",
            background=self.ACCENT,
            troughcolor="#E3E7E4",
            bordercolor="#E3E7E4",
            lightcolor=self.ACCENT,
            darkcolor=self.ACCENT)
        style.configure(
            "App.TCombobox",
            fieldbackground=self.PANEL_BACKGROUND,
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            bordercolor=self.LINE,
            lightcolor=self.LINE,
            darkcolor=self.LINE,
            arrowcolor=self.TEXT_SECONDARY,
            arrowsize=14,
            padding=7)
        combo_normal = rounded_tile(
            40,
            15,
            self.PANEL_BACKGROUND,
            self.LINE,
            self.PANEL_BACKGROUND)
        combo_focus = rounded_tile(
            40,
            15,
            self.PANEL_BACKGROUND,
            self.ACCENT,
            self.PANEL_BACKGROUND)
        style.element_create(
            "AutoAnkiRoundedCombobox.field",
            "image",
            combo_normal,
            ("focus", combo_focus),
            border=(15, 15, 15, 15),
            sticky="nsew")
        style.layout(
            "App.TCombobox",
            [(
                "AutoAnkiRoundedCombobox.field",
                {
                    "sticky": "nsew",
                    "children": [
                        (
                            "Combobox.downarrow",
                            {"side": "right", "sticky": ""}),
                        (
                            "Combobox.padding",
                            {
                                "sticky": "nsew",
                                "children": [(
                                    "Combobox.textarea",
                                    {"sticky": "nsew"})],
                            }),
                    ],
                })])
        style.configure(
            "App.TEntry",
            fieldbackground=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            bordercolor=self.LINE,
            lightcolor=self.LINE,
            darkcolor=self.LINE,
            padding=8)
        style.map(
            "App.TEntry",
            bordercolor=[("focus", self.ACCENT)])
        style.configure(
            "Jobs.Treeview",
            background=self.PANEL_BACKGROUND,
            fieldbackground=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            rowheight=30,
            borderwidth=0,
            font=("DejaVu Sans", 9))
        style.map(
            "Jobs.Treeview",
            background=[("selected", "#D7E9E5")],
            foreground=[("selected", self.TEXT_PRIMARY)])
        style.configure(
            "Jobs.Treeview.Heading",
            background=self.PALE,
            foreground=self.TEXT_PRIMARY,
            relief=tk.FLAT,
            font=("DejaVu Sans", 9, "bold"),
            padding=(8, 7))
        style.map(
            "Jobs.Treeview.Heading",
            background=[("active", "#DDEBE7")])
        style.configure(
            "ValidationProblems.Treeview",
            background=self.PANEL_BACKGROUND,
            fieldbackground=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            rowheight=34,
            borderwidth=0,
            font=("DejaVu Sans", 9))
        style.map(
            "ValidationProblems.Treeview",
            background=[("selected", "#D7E9E5")],
            foreground=[("selected", self.TEXT_PRIMARY)])
        style.configure(
            "ValidationProblems.Treeview.Heading",
            background=self.PALE,
            foreground=self.TEXT_PRIMARY,
            relief=tk.FLAT,
            font=("DejaVu Sans", 9, "bold"),
            padding=(8, 7))
        style.map(
            "ValidationProblems.Treeview.Heading",
            background=[("active", "#DDEBE7")])
        style.configure(
            "TNotebook",
            background=self.WINDOW_BACKGROUND,
            borderwidth=0,
            tabmargins=(0, 0, 0, 0))
        style.configure(
            "TNotebook.Tab",
            padding=(16, 8),
            font=("DejaVu Sans", 10, "bold"),
            borderwidth=0)
        style.map(
            "TNotebook.Tab",
            padding=[
                ("selected", (20, 13)),
                ("!selected", (16, 8))],
            background=[
                ("selected", self.PANEL_BACKGROUND),
                ("active", "#E7E8E4"),
                ("!selected", "#DDE1DD")],
            foreground=[
                ("selected", self.TEXT_PRIMARY),
                ("!selected", self.TEXT_SECONDARY)])

    @staticmethod
    def _clear_text_selection(event):
        """Remove transient highlighting when a text widget loses focus."""
        event.widget.tag_remove(tk.SEL, "1.0", tk.END)

    @staticmethod
    def _clear_entry_selection(event):
        """Remove transient highlighting when an entry loses focus."""
        try:
            event.widget.selection_clear()
        except tk.TclError:
            # A combobox can be destroyed while a delayed focus event is
            # still waiting in Tk's event queue.
            pass

    def _clear_focus_on_background_click(self, event):
        interactive_classes = {
            "Button",
            "Checkbutton",
            "Combobox",
            "Entry",
            "Listbox",
            "Menubutton",
            "Radiobutton",
            "Scale",
            "Scrollbar",
            "Spinbox",
            "TButton",
            "TCheckbutton",
            "TCombobox",
            "TEntry",
            "TMenubutton",
            "TNotebook",
            "TRadiobutton",
            "TScale",
            "TScrollbar",
            "TSpinbox",
            "Treeview",
            "Text",
        }
        try:
            widget_class = event.widget.winfo_class()
        except Exception:
            return
        if widget_class not in interactive_classes:
            self.root.focus_set()

    def _build_widgets(self):
        header = tk.Frame(
            self.root,
            background=self.DARK,
            borderwidth=0)
        header.grid(row=0, column=0, sticky="ew")
        header_inner = tk.Frame(
            header,
            background=self.DARK,
            borderwidth=0)
        header_inner.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=20)
        ttk.Label(
            header_inner,
            text="AUTOANKI",
            style="Subtitle.TLabel").pack(anchor="w")
        ttk.Label(
            header_inner,
            text="Retained knowledge",
            style="Title.TLabel").pack(
                anchor="w",
                pady=(2, 0))

        frame = ttk.Frame(
            self.root,
            padding=(28, 18, 28, 22),
            style="App.TFrame")
        frame.grid(row=1, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.generate_tab = ttk.Frame(
            self.notebook,
            padding=(4, 16, 4, 4),
            style="App.TFrame")
        self.pipeline_tab = ttk.Frame(
            self.notebook,
            padding=(4, 16, 4, 4),
            style="App.TFrame")
        self.advanced_tab = ttk.Frame(
            self.notebook,
            padding=(4, 16, 4, 4),
            style="App.TFrame")
        self.help_tab = ttk.Frame(
            self.notebook,
            padding=(4, 16, 4, 4),
            style="App.TFrame")
        self.notebook.add(
            self.generate_tab,
            text="Generate")
        self.notebook.add(
            self.pipeline_tab,
            text="Card setup")
        self.notebook.add(
            self.advanced_tab,
            text="Advanced")
        self.notebook.add(
            self.help_tab,
            text="Help")

        self._build_generate_tab()

    def _build_generate_tab(self):
        tab = self.generate_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        self.generate_notebook = ttk.Notebook(tab)
        self.generate_notebook.grid(
            row=0,
            column=0,
            sticky="nsew")
        self.manual_generate_tab = ttk.Frame(
            self.generate_notebook,
            padding=(4, 14, 4, 4),
            style="App.TFrame")
        self.from_source_tab = ttk.Frame(
            self.generate_notebook,
            padding=(4, 14, 4, 4),
            style="App.TFrame")
        self.generate_notebook.add(
            self.manual_generate_tab,
            text="Manual Input")
        self.generate_notebook.add(
            self.from_source_tab,
            text="From Source")
        self.generate_notebook.bind(
            "<<NotebookTabChanged>>",
            self._generate_mode_changed,
            add="+")

        self._build_manual_generate_tab()

    def _build_manual_generate_tab(self):
        tab = self.manual_generate_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        editor_surface = RoundedPanel(
            tab,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.WINDOW_BACKGROUND,
            radius=27,
            inset=18)
        editor_surface.grid(row=0, column=0, sticky="nsew")
        editor = editor_surface.interior
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(2, weight=1)

        input_header = ttk.Frame(
            editor,
            style="Panel.TFrame")
        input_header.grid(row=0, column=0, sticky="ew")
        input_header.columnconfigure(0, weight=1)

        ttk.Label(
            input_header,
            text="WORDS AND NOTES",
            style="Section.TLabel").grid(
                row=0,
                column=0,
                sticky="w")
        ttk.Label(
            input_header,
            text="INPUT LANGUAGE",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=1,
                sticky="e",
                padx=(18, 8))
        self.generation_language_box = ttk.Combobox(
            input_header,
            textvariable=self.generation_language,
            values=tuple(
                language.name
                for language in pipeline_store.list_languages()),
            width=language_selector_width(),
            state="readonly",
            style="App.TCombobox")
        self.generation_language_box.grid(
            row=0,
            column=2,
            sticky="e",
            padx=(0, 18))
        self.generation_language_box.bind(
            "<<ComboboxSelected>>",
            self._generation_language_changed,
            add="+")
        self.generation_language_box.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")

        ttk.Label(
            input_header,
            textvariable=self.input_count,
            style="Muted.TLabel").grid(
                row=0,
                column=3,
                sticky="e")

        ttk.Label(
            editor,
            text=(
                "The input uses the selected language's saved card settings. "
                "Each run makes one OpenAI API request."),
            style="Muted.TLabel",
            wraplength=880).grid(
                row=1,
                column=0,
                sticky="w",
                pady=(4, 10))

        input_surface = RoundedPanel(
            editor,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.PANEL_BACKGROUND,
            radius=22,
            inset=5)
        input_surface.grid(row=2, column=0, sticky="nsew")
        input_editor = input_surface.interior
        input_editor.columnconfigure(0, weight=1)
        input_editor.rowconfigure(0, weight=1)

        self.input_text = tk.Text(
            input_editor,
            wrap=tk.WORD,
            undo=True,
            font=("DejaVu Sans", 11),
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            insertbackground=self.TEXT_PRIMARY,
            selectbackground="#D7E9E5",
            selectforeground=self.TEXT_PRIMARY,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=12,
            pady=10)
        input_scrollbar = RoundedScrollbar(
            input_editor,
            command=self.input_text.yview,
            background=self.PANEL_BACKGROUND,
            active=self.ACCENT)
        self.input_text.configure(
            yscrollcommand=input_scrollbar.set)
        self.input_text.grid(row=0, column=0, sticky="nsew")
        input_scrollbar.grid(row=0, column=1, sticky="ns")
        self._restore_manual_input_draft()
        self.input_text.bind(
            "<<Modified>>",
            self._manual_input_changed)
        self.input_text.bind(
            "<FocusOut>",
            self._clear_text_selection,
            add="+")
        self._update_input_count()
        self.input_text.focus_set()

        controls = ttk.Frame(
            tab,
            style="App.TFrame")
        controls.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(16, 0))
        controls.columnconfigure(2, weight=1)

        self.generate_button = ttk.Button(
            controls,
            text="Generate and import",
            command=self.start_generation,
            style="Accent.TButton",
            cursor="hand2")
        self.generate_button.grid(
            row=0,
            column=0,
            sticky="w")

        self.status_dot = tk.Canvas(
            controls,
            width=14,
            height=14,
            background=self.WINDOW_BACKGROUND,
            highlightthickness=0,
            borderwidth=0)
        self.status_dot.grid(
            row=0,
            column=1,
            padx=(14, 7))
        self.status_circle = self.status_dot.create_oval(
            2, 2, 12, 12,
            fill=self.TEXT_SECONDARY,
            outline=self.TEXT_SECONDARY)

        ttk.Label(
            controls,
            textvariable=self.status,
            wraplength=540,
            style="Status.TLabel").grid(
                row=0,
                column=2,
                sticky="w",
                padx=(0, 12))

        self.api_key_button = ttk.Menubutton(
            controls,
            text="API key",
            style="Secondary.TButton",
            cursor="hand2")
        self.api_key_button.grid(
            row=0,
            column=3,
            sticky="e")

        api_key_menu = tk.Menu(
            self.api_key_button,
            tearoff=False)
        api_key_menu.add_command(
            label="Save or replace key…",
            command=self.configure_api_key)
        api_key_menu.add_command(
            label="Forget saved key…",
            command=self.forget_api_key)
        self.api_key_button.configure(menu=api_key_menu)

        self.generation_progress = ttk.Progressbar(
            controls,
            mode="indeterminate",
            style="Generation.Horizontal.TProgressbar")
        self.generation_progress.grid(
            row=1,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(10, 0))
        self.generation_progress.grid_remove()

        output_surface = RoundedPanel(
            tab,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.WINDOW_BACKGROUND,
            radius=20,
            inset=12)
        output_surface.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(16, 0))
        output_panel = output_surface.interior
        output_panel.columnconfigure(1, weight=1)

        ttk.Label(
            output_panel,
            text="LAST OUTPUT",
            style="Section.TLabel").grid(
                row=0,
                column=0,
                sticky="w")
        ttk.Label(
            output_panel,
            textvariable=self.output,
            wraplength=650,
            style="Muted.TLabel").grid(
                row=0,
                column=1,
                sticky="w",
                padx=(14, 0))

    def _build_from_source_tab(self):
        if self._from_source_tab_built:
            return
        self._from_source_tab_built = True
        tab = self.from_source_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        self.source_options = ()
        self.source_options_by_label = {}
        self.source_options_by_key = {}
        self.source_display_labels_by_key = {}
        self.source_selected_label = tk.StringVar()
        self.source_language_label = tk.StringVar()
        preferences = self.gui_preferences
        self.source_chunk_size = tk.StringVar(
            value=preferences["source_chunk_size"])
        self.source_limit_to_prefix = tk.BooleanVar(
            value=preferences["source_limit_to_prefix"])
        self.source_prefix_token_limit = tk.StringVar(
            value=preferences["source_prefix_token_limit"])
        self.source_concurrency = tk.StringVar(
            value=preferences["source_concurrency"])
        self.source_request_stagger_ms = tk.StringVar(
            value=preferences["source_request_stagger_ms"])
        self.source_allow_web_search = tk.BooleanVar(
            value=preferences["source_allow_web_search"])
        self.source_use_source_examples = tk.BooleanVar(
            value=preferences["source_use_source_examples"])
        self.source_card_direction_variables = {
            direction.key: tk.BooleanVar(
                value=(
                    direction.key
                    in preferences["source_card_directions"]))
            for direction in pipeline_store.list_directions()
        }
        self.source_card_direction_summary = tk.StringVar(
            value="Sentence → Meaning")
        self.source_include_context_nuance = tk.BooleanVar(
            value=preferences["source_include_context_nuance"])
        self.source_separate_decks = tk.BooleanVar(
            value=preferences["source_separate_decks"])
        self.source_model_label = tk.StringVar(
            value=dict(SOURCE_MODEL_OPTIONS).get(
                preferences["source_model_key"],
                SOURCE_MODEL_OPTIONS[0][1]))
        self.source_request_protocol_label = tk.StringVar(
            value=dict(SOURCE_PROTOCOL_OPTIONS).get(
                preferences["source_protocol_key"],
                SOURCE_PROTOCOL_OPTIONS[0][1]))
        self.source_reasoning_label = tk.StringVar(
            value=dict(SOURCE_REASONING_OPTIONS).get(
                preferences["source_reasoning_key"],
                SOURCE_REASONING_OPTIONS[0][1]))
        self.source_execution_label = tk.StringVar(
            value=dict(SOURCE_EXECUTION_OPTIONS).get(
                preferences["source_execution_key"],
                SOURCE_EXECUTION_OPTIONS[0][1]))
        self.source_automatic_repair = tk.BooleanVar(
            value=preferences["source_automatic_repair"])
        self.source_context_label = tk.StringVar(
            value=SOURCE_CONTEXT_LABELS.get(
                preferences["source_context_key"],
                SOURCE_CONTEXT_LABELS["sentence"]))
        self._source_context_before_source_examples = (
            preferences["source_non_example_context_key"]
            if preferences["source_non_example_context_key"]
            in SOURCE_CONTEXT_LABELS
            else "sentence")
        self._source_reasoning_before_v8 = (
            preferences["source_reasoning_key"]
            if preferences["source_reasoning_key"]
            in dict(SOURCE_REASONING_OPTIONS)
            else "low")
        self.source_context_description = tk.StringVar()
        self.source_summary = tk.StringVar()
        self.source_deck_notice = tk.StringVar()
        self.source_catalog_status = tk.StringVar()
        self.source_estimate_price = tk.StringVar(
            value="Estimate unavailable")
        self.source_estimate_detail = tk.StringVar(
            value="Select a prepared source to calculate an estimate.")
        self.source_estimate_result = None
        self.source_estimate_generation = 0
        self.source_estimate_pending = set()
        self.source_estimate_polling = False
        self.source_estimate_result_queue = queue.Queue()
        self.source_zero_notice_shown = False
        self.source_paid_authorized = tk.BooleanVar(value=False)
        self.source_action_status = tk.StringVar(value="No source job running.")
        self.source_dispatch_status = tk.StringVar(
            value=(
                "Ready"
                if (
                    self.source_pause_callback is not None
                    and self.source_resume_callback is not None)
                else "Emergency control unavailable"))
        self.source_preview_page_size = tk.StringVar(
            value=preferences["source_preview_page_size"])
        self.source_preview_status = tk.StringVar(
            value="Choose a prepared source, then load its local metadata.")
        self.source_preview_page_data = None
        self.source_preview_by_tree_id = {}
        self._source_jobs_page_built = False

        self.source_notebook = ttk.Notebook(tab)
        self.source_notebook.grid(
            row=0,
            column=0,
            sticky="nsew")
        self.source_generate_page = ttk.Frame(
            self.source_notebook,
            style="App.TFrame")
        self.source_preview_page = ttk.Frame(
            self.source_notebook,
            padding=(4, 14, 4, 4),
            style="App.TFrame")
        self.source_prepare_page = ttk.Frame(
            self.source_notebook,
            style="App.TFrame")
        self.source_codex_page = ttk.Frame(
            self.source_notebook,
            style="App.TFrame")
        self.source_jobs_page = ttk.Frame(
            self.source_notebook,
            padding=(4, 14, 4, 4),
            style="App.TFrame")
        self.source_notebook.add(
            self.source_generate_page,
            text="Generate Deck")
        self.source_notebook.add(
            self.source_preview_page,
            text="Inspect Source")
        self.source_notebook.add(
            self.source_prepare_page,
            text="Prepare File")
        self.source_notebook.add(
            self.source_codex_page,
            text="Codex Retrieve")
        self.source_notebook.add(
            self.source_jobs_page,
            text="Jobs & Failures")
        self.source_notebook.bind(
            "<<NotebookTabChanged>>",
            self._source_page_changed,
            add="+")

        self._build_source_generate_page()
        self._build_source_preview_page()
        self._build_source_prepare_page()
        self._build_source_codex_page()
        self._load_source_catalogue()
        self._update_source_context_description()
        self._update_source_exclusion_controls()
        self._update_source_generate_button_state()
        self._restore_source_page_selection()
        self._install_source_preference_tracking()

    def _ensure_from_source_tab_built(self):
        if not getattr(self, "_from_source_tab_built", False):
            self._build_from_source_tab()

    def _restore_source_page_selection(self):
        source_tabs = {
            "generate": self.source_generate_page,
            "inspect": self.source_preview_page,
            "prepare": self.source_prepare_page,
            "codex": self.source_codex_page,
            "jobs": self.source_jobs_page,
        }
        target = source_tabs.get(
            self.gui_preferences.get("source_tab"))
        if target is None:
            return
        if target == self.source_jobs_page:
            self._ensure_source_jobs_page_built()
        self.source_notebook.select(target)

    def _source_header(self, parent, title, body):
        surface = RoundedPanel(
            parent,
            fill=self.PALE,
            outline="#C8DDD8",
            background=self.WINDOW_BACKGROUND,
            radius=20,
            inset=14)
        surface.grid(
            row=0,
            column=0,
            sticky="ew")
        header = surface.interior
        header.columnconfigure(0, weight=1)
        tk.Label(
            header,
            text=title,
            background=self.PALE,
            foreground=self.TEXT_PRIMARY,
            font=("DejaVu Sans", 11, "bold")).grid(
                row=0,
                column=0,
                sticky="w")
        tk.Label(
            header,
            text=body,
            background=self.PALE,
            foreground=self.TEXT_SECONDARY,
            font=("DejaVu Sans", 9),
            wraplength=1100,
            anchor="w",
            justify="left").grid(
                row=1,
                column=0,
                sticky="ew",
                pady=(4, 0))
        return surface

    def _build_source_generate_page(self):
        page = self.source_generate_page
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)
        viewport = ScrollableFrame(
            page,
            background=self.WINDOW_BACKGROUND,
            frame_style="App.TFrame")
        viewport.grid(row=0, column=0, sticky="nsew")
        self.source_generate_viewport = viewport
        content = viewport.content
        content.columnconfigure(0, weight=1)

        self._source_header(
            content,
            "GENERATE FROM A PREPARED SOURCE",
            (
                "The source is already tokenized in first-occurrence order. "
                "Card detail and response languages come from Card setup; "
                "context is deduplicated before it is sent."
            ))

        config_surface = RoundedPanel(
            content,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.WINDOW_BACKGROUND,
            radius=24,
            inset=18)
        config_surface.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(10, 0))
        config = config_surface.interior
        config.columnconfigure(1, weight=1)

        ttk.Label(
            config,
            text="PREPARED SOURCE",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                padx=(0, 12))
        self.source_selector = ttk.Combobox(
            config,
            textvariable=self.source_selected_label,
            state="readonly",
            style="App.TCombobox",
            width=42)
        self.source_selector.grid(
            row=0,
            column=1,
            sticky="ew")
        self.source_selector.bind(
            "<<ComboboxSelected>>",
            self._source_selection_changed,
            add="+")
        self.source_selector.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")
        ttk.Button(
            config,
            text="Refresh",
            command=self._refresh_source_catalogue,
            style="Secondary.TButton",
            cursor="hand2").grid(
                row=0,
                column=2,
                sticky="e",
                padx=(10, 0))

        ttk.Label(
            config,
            textvariable=self.source_summary,
            style="Muted.TLabel",
            wraplength=950).grid(
                row=1,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(7, 14))

        ttk.Label(
            config,
            text="SOURCE LANGUAGE / REGISTER",
            style="FieldLabel.TLabel").grid(
                row=2,
                column=0,
                sticky="w",
                padx=(0, 12))
        self.source_language_selector = ttk.Combobox(
            config,
            textvariable=self.source_language_label,
            values=tuple(
                language.name
                for language in pipeline_store.list_languages()),
            state="readonly",
            style="App.TCombobox",
            width=language_selector_width())
        self.source_language_selector.grid(
            row=2,
            column=1,
            columnspan=2,
            sticky="ew")
        self.source_language_selector.bind(
            "<<ComboboxSelected>>",
            self._source_language_changed,
            add="+")
        self.source_language_selector.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")

        ttk.Label(
            config,
            text="CONTEXT PER WORD",
            style="FieldLabel.TLabel").grid(
                row=3,
                column=0,
                sticky="w",
                padx=(0, 12))
        context_box = ttk.Combobox(
            config,
            textvariable=self.source_context_label,
            values=tuple(
                label
                for _key, label, _description
                in SOURCE_CONTEXT_OPTIONS),
            state="readonly",
            style="App.TCombobox",
            width=42)
        self.source_context_selector = context_box
        context_box.grid(
            row=3,
            column=1,
            columnspan=2,
            sticky="ew")
        context_box.bind(
            "<<ComboboxSelected>>",
            self._source_context_changed,
            add="+")
        context_box.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")
        ttk.Label(
            config,
            textvariable=self.source_context_description,
            style="Muted.TLabel",
            wraplength=950).grid(
                row=4,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(7, 14))

        source_card_options = ttk.Frame(
            config,
            style="Panel.TFrame")
        source_card_options.grid(
            row=7,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(0, 14))
        source_card_options.columnconfigure(1, weight=1)
        ttk.Label(
            source_card_options,
            text="CARD TYPES",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                padx=(0, 12))
        self.source_card_type_button = ttk.Menubutton(
            source_card_options,
            textvariable=self.source_card_direction_summary,
            style="Secondary.TButton",
            cursor="hand2")
        self.source_card_type_button.grid(
            row=0,
            column=1,
            sticky="w")
        source_card_type_menu = tk.Menu(
            self.source_card_type_button,
            tearoff=False)
        for direction in pipeline_store.list_directions():
            source_card_type_menu.add_checkbutton(
                label=direction.name,
                variable=self.source_card_direction_variables[direction.key],
                command=self._source_card_options_changed)
        self.source_card_type_button.configure(
            menu=source_card_type_menu)
        self.source_include_context_nuance_check = ttk.Checkbutton(
            source_card_options,
            text="Include cultural / historical sentence nuance",
            variable=self.source_include_context_nuance,
            command=self._source_card_options_changed,
            style="Panel.TCheckbutton")
        self.source_include_context_nuance_check.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(9, 0))
        ttk.Checkbutton(
            source_card_options,
            text="Package each selected card type in a separate deck",
            variable=self.source_separate_decks,
            command=self._source_card_options_changed,
            style="Panel.TCheckbutton").grid(
                row=2,
                column=0,
                columnspan=2,
                sticky="w",
                pady=(7, 0))

        self.source_use_source_examples_check = ttk.Checkbutton(
            config,
            text="Use source sentences for Sentence → Meaning cards",
            variable=self.source_use_source_examples,
            command=self._source_example_setting_changed,
            style="Panel.TCheckbutton")
        self.source_use_source_examples_check.grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="w")
        ttk.Label(
            config,
            text=(
                "Create one sentence card from each exact source sentence. "
                "The original is inserted locally; the model returns its "
                "English translation and, optionally, sentence-level nuance. "
                "Word cards request lexical fields separately."
            ),
            style="Muted.TLabel",
            wraplength=950).grid(
                row=6,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(5, 14))

        prefix_controls = ttk.Frame(
            config,
            style="Panel.TFrame")
        prefix_controls.grid(
            row=8,
            column=0,
            columnspan=3,
            sticky="ew")
        self.source_limit_to_prefix_check = ttk.Checkbutton(
            prefix_controls,
            text="Generate only from the first",
            variable=self.source_limit_to_prefix,
            command=self._source_prefix_setting_changed,
            style="Panel.TCheckbutton")
        self.source_limit_to_prefix_check.grid(
            row=0,
            column=0,
            sticky="w")
        self.source_prefix_token_entry = ttk.Entry(
            prefix_controls,
            textvariable=self.source_prefix_token_limit,
            width=10,
            style="App.TEntry")
        self.source_prefix_token_entry.grid(
            row=0,
            column=1,
            sticky="w",
            padx=(8, 8))
        self.source_prefix_token_entry.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")
        ttk.Label(
            prefix_controls,
            text="token occurrences (running words) of the source",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=2,
                sticky="w")
        ttk.Label(
            config,
            text=(
                "This is a true text prefix, including repeated words. "
                "AutoAnki then deduplicates the vocabulary first encountered "
                "inside that prefix, so 100 running words may produce fewer "
                "than 100 cards."
            ),
            style="Muted.TLabel",
            wraplength=950).grid(
                row=9,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(5, 14))

        ttk.Label(
            config,
            text="WORDS PER MODEL REQUEST",
            style="FieldLabel.TLabel").grid(
                row=10,
                column=0,
                sticky="w",
                padx=(0, 12))
        self.source_chunk_entry = ttk.Entry(
            config,
            textvariable=self.source_chunk_size,
            width=14,
            style="App.TEntry")
        self.source_chunk_entry.grid(
            row=10,
            column=1,
            sticky="w")
        self.source_chunk_entry.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")
        ttk.Label(
            config,
            text=(
                "Smaller chunks repeat the instructions more often; larger "
                "chunks reduce prompt overhead but produce larger responses."
            ),
            style="Muted.TLabel",
            wraplength=570).grid(
                row=10,
                column=2,
                sticky="w",
                padx=(12, 0))

        mode_controls = ttk.Frame(
            config,
            style="Panel.TFrame")
        mode_controls.grid(
            row=11,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(14, 0))
        for column in range(2):
            mode_controls.columnconfigure(column, weight=1)
        for index, (heading, variable, values, callback) in enumerate((
                (
                    "MODEL",
                    self.source_model_label,
                    tuple(label for _key, label in SOURCE_MODEL_OPTIONS),
                    self._source_model_changed,
                ),
                (
                    "REQUEST PROTOCOL",
                    self.source_request_protocol_label,
                    tuple(label for _key, label in SOURCE_PROTOCOL_OPTIONS),
                    self._source_protocol_changed,
                ),
                (
                    "REASONING",
                    self.source_reasoning_label,
                    tuple(label for _key, label in SOURCE_REASONING_OPTIONS),
                    self._source_reasoning_changed,
                ),
                (
                    "PROCESSING",
                    self.source_execution_label,
                    tuple(label for _key, label in SOURCE_EXECUTION_OPTIONS),
                    self._source_execution_mode_changed,
                ),
        )):
            column = index % 2
            label_row = (index // 2) * 2
            ttk.Label(
                mode_controls,
                text=heading,
                style="FieldLabel.TLabel").grid(
                    row=label_row,
                    column=column,
                    sticky="w",
                    padx=(0 if column == 0 else 16, 8),
                    pady=(10 if label_row else 0, 0))
            selector = ttk.Combobox(
                mode_controls,
                textvariable=variable,
                values=values,
                state="readonly",
                style="App.TCombobox",
                width=29)
            selector.grid(
                row=label_row + 1,
                column=column,
                sticky="ew",
                padx=(0 if column == 0 else 16, 0),
                pady=(4, 0))
            selector.bind(
                "<<ComboboxSelected>>",
                callback,
                add="+")
            selector.bind(
                "<FocusOut>",
                self._clear_entry_selection,
                add="+")
            if heading == "REASONING":
                self.source_reasoning_selector = selector
            elif heading == "MODEL":
                self.source_model_selector = selector
            elif heading == "REQUEST PROTOCOL":
                self.source_protocol_selector = selector
            elif heading == "PROCESSING":
                self.source_execution_selector = selector

        ttk.Label(
            config,
            text=(
                "Compact v10 (recommended) keeps provider sentences plain, "
                "adds term emphasis locally, and can discard only provably "
                "unusable optional senses before strict validation. Compact "
                "v9 and legacy v8 remain one-click rollbacks. GPT-5.4 mini "
                "is the supported model. Low reasoning with the 30-word "
                "default is the tested recommendation; None is cheaper for "
                "small chunks, while larger chunks can omit coverage or "
                "senses. Economy submits an asynchronous OpenAI "
                "Batch at half token pricing; it can take up to 24 hours and "
                "is collected from Jobs & Failures."
            ),
            style="Muted.TLabel",
            wraplength=950).grid(
                row=12,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(7, 0))

        self.source_automatic_repair_check = ttk.Checkbutton(
            config,
            text="Automatic repair",
            variable=self.source_automatic_repair,
            command=self._source_automatic_repair_changed,
            style="Panel.TCheckbutton")
        self.source_automatic_repair_check.grid(
            row=13,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(14, 0))
        ttk.Label(
            config,
            text=(
                "Standard processing only. Repairs replace the smallest "
                "safe scope first, with no more than three additional paid "
                "micro-repair calls per failed request."
            ),
            style="Muted.TLabel",
            wraplength=950).grid(
                row=14,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(5, 0))

        rate_controls = ttk.Frame(
            config,
            style="Panel.TFrame")
        rate_controls.grid(
            row=15,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(14, 0))
        ttk.Label(
            rate_controls,
            text="MAX PARALLEL REQUESTS",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                padx=(0, 8))
        concurrency_entry = ttk.Entry(
            rate_controls,
            textvariable=self.source_concurrency,
            width=7,
            style="App.TEntry")
        concurrency_entry.grid(
            row=0,
            column=1,
            sticky="w")
        concurrency_entry.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")
        self.source_concurrency_entry = concurrency_entry
        ttk.Label(
            rate_controls,
            text="MINIMUM START STAGGER (MS)",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=2,
                sticky="w",
                padx=(24, 8))
        stagger_entry = ttk.Entry(
            rate_controls,
            textvariable=self.source_request_stagger_ms,
            width=9,
            style="App.TEntry")
        stagger_entry.grid(
            row=0,
            column=3,
            sticky="w")
        stagger_entry.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")
        self.source_stagger_entry = stagger_entry
        self.source_rate_notice_label = ttk.Label(
            config,
            text=(
                "The default starts eight workers, with requests launched no "
                "faster than the stagger permits. Account limits vary; the "
                "coordinator observes OpenAI rate-limit responses and backs "
                "off automatically. Connection/time-out, HTTP 429, and HTTP "
                "5xx failures may retry without asking; retry costs are not "
                "included in the estimate."
            ),
            style="Muted.TLabel",
            wraplength=950)
        self.source_rate_notice_label.grid(
                row=16,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(7, 0))

        self.source_web_search_check = ttk.Checkbutton(
            config,
            text=(
                "Allow one web search per request when the meaning is unclear"
            ),
            variable=self.source_allow_web_search,
            command=self._schedule_source_estimate,
            style="Panel.TCheckbutton")
        self.source_web_search_check.grid(
                row=17,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(14, 0))
        ttk.Label(
            config,
            text=(
                "The model decides whether a search is necessary. The cost "
                "range includes at most one search call and a conservative "
                "search-content allowance for every request."
            ),
            style="Muted.TLabel",
            wraplength=950).grid(
                row=18,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(5, 0))

        estimate_surface = RoundedPanel(
            content,
            fill=self.PALE,
            outline="#C8DDD8",
            background=self.WINDOW_BACKGROUND,
            radius=22,
            inset=16)
        estimate_surface.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(10, 0))
        estimate = estimate_surface.interior
        estimate.columnconfigure(0, weight=1)
        self.source_estimate_heading_label = tk.Label(
            estimate,
            text="ESTIMATED OPENAI COST",
            background=self.PALE,
            foreground=self.TEXT_SECONDARY,
            font=("DejaVu Sans", 8, "bold"))
        self.source_estimate_heading_label.grid(
            row=0,
            column=0,
            sticky="w")
        tk.Label(
            estimate,
            textvariable=self.source_estimate_price,
            background=self.PALE,
            foreground=self.TEXT_PRIMARY,
            font=("DejaVu Sans", 19, "bold")).grid(
                row=1,
                column=0,
                sticky="w",
                pady=(3, 0))
        tk.Label(
            estimate,
            textvariable=self.source_estimate_detail,
            background=self.PALE,
            foreground=self.TEXT_SECONDARY,
            font=("DejaVu Sans", 9),
            wraplength=1000,
            justify="left").grid(
                row=2,
                column=0,
                sticky="ew",
                pady=(5, 0))

        actions_surface = RoundedPanel(
            content,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.WINDOW_BACKGROUND,
            radius=22,
            inset=16)
        actions_surface.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(10, 0))
        actions = actions_surface.interior
        actions.columnconfigure(0, weight=1)
        ttk.Label(
            actions,
            textvariable=self.source_deck_notice,
            style="Section.TLabel",
            wraplength=850).grid(
                row=0,
                column=0,
                columnspan=2,
                sticky="w")
        ttk.Label(
            actions,
            text=(
                "The imported source deck is retained as-is. AutoAnki will "
                "not move its cards into another deck or delete it."
            ),
            style="Muted.TLabel",
            wraplength=920).grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="w",
                pady=(4, 12))
        self.source_authorization_check = ttk.Checkbutton(
            actions,
            text=(
                "I authorise the paid OpenAI requests shown in this estimate"
            ),
            variable=self.source_paid_authorized,
            command=self._update_source_generate_button_state,
            style="Panel.TCheckbutton")
        self.source_authorization_check.grid(
                row=2,
                column=0,
                sticky="w")
        self.source_generate_button = ttk.Button(
            actions,
            text="Generate and import source deck",
            command=self._start_source_generation,
            state=tk.DISABLED,
            style="Accent.TButton",
            cursor="hand2")
        self.source_generate_button.grid(
            row=2,
            column=1,
            sticky="e",
            padx=(14, 0))
        ttk.Label(
            content,
            textvariable=self.source_action_status,
            style="Status.TLabel",
            wraplength=1080).grid(
                row=4,
                column=0,
                sticky="w",
                pady=(10, 0))

        for variable in (
                self.source_language_label,
                self.source_chunk_size,
                self.source_prefix_token_limit,
                self.source_concurrency,
                self.source_request_stagger_ms):
            variable.trace_add(
                "write",
                self._schedule_source_estimate)
        self._sync_source_prefix_control()
        self._source_protocol_changed()
        self._source_execution_mode_changed()
        viewport.bind_mousewheel_tree()

    def _build_source_preview_page(self):
        page = self.source_preview_page
        page.columnconfigure(0, weight=1)
        page.rowconfigure(2, weight=1)

        preview_header = tk.Frame(
            page,
            background=self.PALE,
            highlightbackground="#C8DDD8",
            highlightthickness=1,
            padx=14,
            pady=9)
        preview_header.grid(
            row=0,
            column=0,
            sticky="ew")
        tk.Label(
            preview_header,
            text="INSPECT A PREPARED SOURCE",
            background=self.PALE,
            foreground=self.TEXT_PRIMARY,
            font=("DejaVu Sans", 11, "bold")).pack(
                anchor="w")
        tk.Label(
            preview_header,
            text=(
                "Browse first-occurrence order, chapter or section, and "
                "retained surrounding sentences. This local read never "
                "calls OpenAI or Anki."
            ),
            background=self.PALE,
            foreground=self.TEXT_SECONDARY,
            font=("DejaVu Sans", 9),
            wraplength=1100,
            anchor="w",
            justify="left").pack(
                anchor="w",
                pady=(3, 0))

        controls = tk.Frame(
            page,
            background=self.PANEL_BACKGROUND,
            highlightbackground=self.LINE,
            highlightthickness=1,
            padx=13,
            pady=9)
        controls.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(10, 0))
        controls.columnconfigure(1, weight=1)

        ttk.Label(
            controls,
            text="PREPARED SOURCE",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                padx=(0, 12))
        self.source_preview_selector = ttk.Combobox(
            controls,
            textvariable=self.source_selected_label,
            state="readonly",
            style="App.TCombobox",
            width=42)
        self.source_preview_selector.grid(
            row=0,
            column=1,
            sticky="ew")
        self.source_preview_selector.bind(
            "<<ComboboxSelected>>",
            self._source_preview_selection_changed,
            add="+")
        self.source_preview_selector.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")

        ttk.Label(
            controls,
            text="ROWS PER PAGE",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=2,
                sticky="w",
                padx=(18, 8))
        preview_page_entry = ttk.Entry(
            controls,
            textvariable=self.source_preview_page_size,
            width=9,
            style="App.TEntry")
        preview_page_entry.grid(
            row=0,
            column=3,
            sticky="w")
        preview_page_entry.bind(
            "<Return>",
            self._reload_source_preview_from_start,
            add="+")
        preview_page_entry.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")
        self.source_preview_load_button = ttk.Button(
            controls,
            text="Load page",
            command=self._reload_source_preview_from_start,
            state=(
                tk.NORMAL
                if self.source_preview_loader is not None
                else tk.DISABLED),
            style="CompactSecondary.TButton",
            cursor="hand2")
        self.source_preview_load_button.grid(
            row=0,
            column=4,
            sticky="e",
            padx=(10, 0))

        ttk.Label(
            controls,
            text=(
                "A page contains local metadata only. The default 500 rows "
                "matches the default OpenAI chunk size, but changing it does "
                "not generate cards or alter the source."
            ),
            style="Muted.TLabel",
            wraplength=1000).grid(
                row=1,
                column=0,
                columnspan=5,
                sticky="w",
                pady=(7, 0))

        preview_body = tk.Frame(
            page,
            background=self.WINDOW_BACKGROUND)
        preview_body.grid(
            row=2,
            column=0,
            sticky="nsew",
            pady=(10, 0))
        preview_body.columnconfigure(0, weight=3)
        preview_body.columnconfigure(1, weight=2)
        preview_body.rowconfigure(0, weight=1)

        table = tk.Frame(
            preview_body,
            background=self.PANEL_BACKGROUND,
            highlightbackground=self.LINE,
            highlightthickness=1,
            padx=8,
            pady=8)
        table.grid(
            row=0,
            column=0,
            sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        columns = ("rank", "term", "section")
        self.source_preview_tree = ttk.Treeview(
            table,
            columns=columns,
            show="headings",
            selectmode="browse",
            height=5,
            style="Jobs.Treeview")
        self.source_preview_tree.heading("rank", text="First-seen rank")
        self.source_preview_tree.heading("term", text="Word")
        self.source_preview_tree.heading(
            "section",
            text="Chapter / section")
        self.source_preview_tree.column(
            "rank",
            width=105,
            minwidth=90,
            stretch=False,
            anchor="e")
        self.source_preview_tree.column(
            "term",
            width=165,
            minwidth=130,
            stretch=False)
        self.source_preview_tree.column(
            "section",
            width=320,
            minwidth=260,
            stretch=True)
        preview_table_scrollbar = RoundedScrollbar(
            table,
            command=self.source_preview_tree.yview,
            background=self.PANEL_BACKGROUND,
            active=self.ACCENT)
        self.source_preview_tree.configure(
            yscrollcommand=preview_table_scrollbar.set)
        self.source_preview_tree.grid(
            row=0,
            column=0,
            sticky="nsew")
        preview_table_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
            padx=(7, 0))
        self.source_preview_tree.bind(
            "<<TreeviewSelect>>",
            self._source_preview_item_selected,
            add="+")

        detail = tk.Frame(
            preview_body,
            background=self.PANEL_BACKGROUND,
            highlightbackground=self.LINE,
            highlightthickness=1,
            padx=8,
            pady=8)
        detail.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(10, 0))
        detail.columnconfigure(0, weight=1)
        detail.rowconfigure(1, weight=1)
        self.source_preview_detail_title = tk.StringVar(
            value="SELECT A WORD TO INSPECT ITS RETAINED CONTEXT")
        ttk.Label(
            detail,
            textvariable=self.source_preview_detail_title,
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                padx=7,
                pady=(4, 5))
        self.source_preview_detail = tk.Text(
            detail,
            height=4,
            width=40,
            wrap=tk.WORD,
            font=("DejaVu Sans", 10),
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            selectbackground="#D7E9E5",
            selectforeground=self.TEXT_PRIMARY,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=8,
            pady=6,
            state=tk.DISABLED)
        self.source_preview_detail.tag_configure(
            "heading",
            foreground=self.ACCENT,
            font=("DejaVu Sans", 9, "bold"),
            spacing1=7,
            spacing3=3)
        self.source_preview_detail.tag_configure(
            "current",
            foreground=self.TEXT_PRIMARY,
            font=("DejaVu Sans", 10, "bold"),
            lmargin1=8,
            lmargin2=8,
            spacing3=4)
        self.source_preview_detail.tag_configure(
            "context",
            foreground=self.TEXT_PRIMARY,
            lmargin1=8,
            lmargin2=8,
            spacing3=4)
        preview_detail_scrollbar = RoundedScrollbar(
            detail,
            command=self.source_preview_detail.yview,
            background=self.PANEL_BACKGROUND,
            active=self.ACCENT)
        self.source_preview_detail.configure(
            yscrollcommand=preview_detail_scrollbar.set)
        self.source_preview_detail.grid(
            row=1,
            column=0,
            sticky="nsew")
        preview_detail_scrollbar.grid(
            row=1,
            column=1,
            sticky="ns")
        self.source_preview_detail.bind(
            "<FocusOut>",
            self._clear_text_selection,
            add="+")

        navigation = ttk.Frame(page, style="App.TFrame")
        navigation.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(10, 0))
        navigation.columnconfigure(2, weight=1)
        self.source_preview_previous_button = ttk.Button(
            navigation,
            text="Previous page",
            command=self._source_preview_previous_page,
            state=tk.DISABLED,
            style="CompactSecondary.TButton",
            cursor="hand2")
        self.source_preview_previous_button.grid(
            row=0,
            column=0,
            sticky="w")
        self.source_preview_next_button = ttk.Button(
            navigation,
            text="Next page",
            command=self._source_preview_next_page,
            state=tk.DISABLED,
            style="CompactSecondary.TButton",
            cursor="hand2")
        self.source_preview_next_button.grid(
            row=0,
            column=1,
            sticky="w",
            padx=(8, 0))
        ttk.Label(
            navigation,
            textvariable=self.source_preview_status,
            style="Status.TLabel",
            wraplength=850).grid(
                row=0,
                column=2,
                sticky="e",
                padx=(14, 0))

    def _build_source_prepare_page(self):
        page = self.source_prepare_page
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)
        viewport = ScrollableFrame(
            page,
            background=self.WINDOW_BACKGROUND,
            frame_style="App.TFrame")
        viewport.grid(row=0, column=0, sticky="nsew")
        content = viewport.content
        content.columnconfigure(0, weight=1)
        self._source_header(
            content,
            "PREPARE A PDF OR TEXT FILE",
            (
                "Extract and tokenize a local source without calling OpenAI "
                "or creating cards. When preparation succeeds, it appears in "
                "the Generate Deck source list."
            ))

        self.source_file_path = tk.StringVar()
        self.source_file_title = tk.StringVar()
        self.source_file_language = tk.StringVar(
            value=(
                self.gui_preferences["source_file_language"]
                if self.gui_preferences["source_file_language"]
                in PREPARABLE_SOURCE_LANGUAGE_NAMES
                else "Classical Chinese (Warring States)"))
        self.source_file_use_gpu = tk.BooleanVar(
            value=self.gui_preferences["source_file_use_gpu"])
        form_surface = RoundedPanel(
            content,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.WINDOW_BACKGROUND,
            radius=24,
            inset=18)
        form_surface.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(10, 0))
        form = form_surface.interior
        form.columnconfigure(1, weight=1)

        ttk.Label(
            form,
            text="PDF OR TEXT FILE",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                padx=(0, 12))
        file_entry = ttk.Entry(
            form,
            textvariable=self.source_file_path,
            style="App.TEntry")
        file_entry.grid(
            row=0,
            column=1,
            sticky="ew")
        file_entry.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")
        ttk.Button(
            form,
            text="Choose file…",
            command=self._choose_source_file,
            style="Secondary.TButton",
            cursor="hand2").grid(
                row=0,
                column=2,
                padx=(10, 0))

        ttk.Label(
            form,
            text="SOURCE TITLE",
            style="FieldLabel.TLabel").grid(
                row=1,
                column=0,
                sticky="w",
                pady=(12, 0),
                padx=(0, 12))
        title_entry = ttk.Entry(
            form,
            textvariable=self.source_file_title,
            style="App.TEntry")
        title_entry.grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="ew",
            pady=(12, 0))
        title_entry.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")

        ttk.Label(
            form,
            text="LANGUAGE / ERA",
            style="FieldLabel.TLabel").grid(
                row=2,
                column=0,
                sticky="w",
                pady=(12, 0),
                padx=(0, 12))
        language_box = ttk.Combobox(
            form,
            textvariable=self.source_file_language,
            values=PREPARABLE_SOURCE_LANGUAGE_NAMES,
            state="readonly",
            width=language_selector_width(),
            style="App.TCombobox")
        language_box.grid(
            row=2,
            column=1,
            columnspan=2,
            sticky="ew",
            pady=(12, 0))
        language_box.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")
        ttk.Checkbutton(
            form,
            text=(
                "Use the NVIDIA GPU when optional CUDA tokenizer support "
                "is installed"
            ),
            variable=self.source_file_use_gpu,
            style="Panel.TCheckbutton").grid(
                row=3,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(14, 0))
        ttk.Label(
            form,
            text=(
                "GPU support is opportunistic: preparation falls back to "
                "the CPU rather than failing when CUDA is unavailable. "
                "The GPU option applies to Classical Chinese; Middle and "
                "Old English use the built-in CPU tokenizer."
            ),
            style="Muted.TLabel",
            wraplength=920).grid(
                row=4,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(5, 12))
        action_row = ttk.Frame(form, style="Panel.TFrame")
        action_row.grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="ew")
        action_row.columnconfigure(0, weight=1)
        ttk.Label(
            action_row,
            text="No OpenAI request or Anki import occurs in this step.",
            style="Muted.TLabel").grid(
                row=0,
                column=0,
                sticky="w")
        self.source_prepare_button = ttk.Button(
            action_row,
            text="Prepare source",
            command=self._start_source_preparation,
            state=(
                tk.NORMAL
                if self.source_prepare_callback is not None
                else tk.DISABLED),
            style="Accent.TButton",
            cursor="hand2")
        self.source_prepare_button.grid(
            row=0,
            column=1,
            sticky="e",
            padx=(12, 0))
        if self.source_prepare_callback is None:
            ttk.Label(
                content,
                text=(
                    "File preparation is not connected in this build yet; "
                    "the controls are safe to inspect."
                ),
                style="Status.TLabel").grid(
                    row=2,
                    column=0,
                    sticky="w",
                    pady=(10, 0))
        viewport.bind_mousewheel_tree()

    def _build_source_codex_page(self):
        page = self.source_codex_page
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)
        viewport = ScrollableFrame(
            page,
            background=self.WINDOW_BACKGROUND,
            frame_style="App.TFrame")
        viewport.grid(row=0, column=0, sticky="nsew")
        content = viewport.content
        content.columnconfigure(0, weight=1)
        self._source_header(
            content,
            "ASK CODEX TO RETRIEVE A SOURCE",
            (
                "Name the prepared source, then describe the exact work, "
                "edition, and retrieval requirements below. Codex does "
                "nothing until you review the request, tick the authorization "
                "box, and press Retrieve and prepare."
            ))

        self.source_codex_language = tk.StringVar(
            value=(
                self.gui_preferences["source_codex_language"]
                if self.gui_preferences["source_codex_language"]
                in PREPARABLE_SOURCE_LANGUAGE_NAMES
                else "Classical Chinese (Warring States)"))
        self.source_codex_title = tk.StringVar()
        self.source_codex_authorized = tk.BooleanVar(value=False)
        request_surface = RoundedPanel(
            content,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.WINDOW_BACKGROUND,
            radius=24,
            inset=16)
        request_surface.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(10, 0))
        request_panel = request_surface.interior
        request_panel.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(
            request_panel,
            style="Panel.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(0, weight=1)
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(
            toolbar,
            text="RETRIEVAL REQUEST",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w")
        ttk.Label(
            toolbar,
            text="LANGUAGE / ERA",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=1,
                sticky="e",
                padx=(16, 8))
        codex_language_box = ttk.Combobox(
            toolbar,
            textvariable=self.source_codex_language,
            values=PREPARABLE_SOURCE_LANGUAGE_NAMES,
            state="readonly",
            width=language_selector_width(),
            style="App.TCombobox")
        codex_language_box.grid(
            row=0,
            column=2,
            sticky="e")
        codex_language_box.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")
        ttk.Label(
            toolbar,
            text="SOURCE TITLE",
            style="FieldLabel.TLabel").grid(
                row=1,
                column=0,
                sticky="w",
                pady=(10, 0))
        codex_title_entry = ttk.Entry(
            toolbar,
            textvariable=self.source_codex_title,
            style="App.TEntry")
        codex_title_entry.grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="ew",
            pady=(10, 0))
        codex_title_entry.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")

        text_surface = RoundedPanel(
            request_panel,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.PANEL_BACKGROUND,
            radius=20,
            inset=5)
        text_surface.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(5, 0))
        ttk.Label(
            request_panel,
            text="SOURCE DESCRIPTION / RETRIEVAL REQUIREMENTS (REQUIRED)",
            style="FieldLabel.TLabel").grid(
                row=2,
                column=0,
                sticky="w",
                pady=(14, 0))
        text_editor = text_surface.interior
        text_editor.columnconfigure(0, weight=1)
        self.source_codex_text = tk.Text(
            text_editor,
            height=11,
            wrap=tk.WORD,
            undo=True,
            font=("DejaVu Sans", 10),
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            insertbackground=self.TEXT_PRIMARY,
            selectbackground="#D7E9E5",
            selectforeground=self.TEXT_PRIMARY,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=12,
            pady=10)
        codex_scrollbar = RoundedScrollbar(
            text_editor,
            command=self.source_codex_text.yview,
            background=self.PANEL_BACKGROUND,
            active=self.ACCENT)
        self.source_codex_text.configure(
            yscrollcommand=codex_scrollbar.set)
        self.source_codex_text.grid(row=0, column=0, sticky="nsew")
        codex_scrollbar.grid(row=0, column=1, sticky="ns")
        self.source_codex_text.bind(
            "<FocusOut>",
            self._clear_text_selection,
            add="+")

        ttk.Label(
            request_panel,
            text=(
                "Describe what Codex should retrieve—for example the desired "
                "edition, original script or language, chapter range, "
                "acceptable sources, completeness requirements, and anything "
                "that must be excluded."
            ),
            style="Muted.TLabel",
            wraplength=980).grid(
                row=4,
                column=0,
                sticky="w",
                pady=(6, 0))

        ttk.Checkbutton(
            request_panel,
            text=(
                "Allow Codex to search for and download source text for "
                "this request"
            ),
            variable=self.source_codex_authorized,
            command=self._update_source_codex_button_state,
            style="Panel.TCheckbutton").grid(
                row=5,
                column=0,
                sticky="w",
                pady=(14, 0))
        action_row = ttk.Frame(
            request_panel,
            style="Panel.TFrame")
        action_row.grid(
            row=6,
            column=0,
            sticky="ew",
            pady=(10, 0))
        action_row.columnconfigure(0, weight=1)
        ttk.Label(
            action_row,
            text=(
                "This retrieves and tokenizes a source only. It never starts "
                "OpenAI card generation."
            ),
            style="Muted.TLabel",
            wraplength=800).grid(
                row=0,
                column=0,
                sticky="w")
        self.source_codex_button = ttk.Button(
            action_row,
            text="Retrieve and prepare",
            command=self._start_source_codex_retrieval,
            state=tk.DISABLED,
            style="Accent.TButton",
            cursor="hand2")
        self.source_codex_button.grid(
            row=0,
            column=1,
            sticky="e",
            padx=(12, 0))
        viewport.bind_mousewheel_tree()

    def _build_source_jobs_page(self):
        if self._source_jobs_page_built:
            return
        self._source_jobs_page_built = True
        tab = self.source_jobs_page
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        viewport = ScrollableFrame(
            tab,
            background=self.WINDOW_BACKGROUND,
            frame_style="App.TFrame")
        viewport.grid(row=0, column=0, sticky="nsew")
        self.source_jobs_viewport = viewport
        page = viewport.content
        page.columnconfigure(0, weight=1)
        self._source_header(
            page,
            "REQUEST JOBS AND FAILURES",
            (
                "Each expandable Job heading is one deck-generation run. "
                "Inside it, OpenAI request chunks appear in order and the "
                "Deck packaging/import stage appears last. Completed chunks "
                "are retained; invalid responses remain stopped until you "
                "inspect and explicitly retry them."
            ))

        table_surface = RoundedPanel(
            page,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.WINDOW_BACKGROUND,
            radius=22,
            inset=10)
        table_surface.grid(
            row=1,
            column=0,
            sticky="nsew",
            pady=(10, 0))
        table = table_surface.interior
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        columns = SOURCE_JOB_COLUMNS
        self.source_jobs_tree = ttk.Treeview(
            table,
            columns=columns,
            show=("tree", "headings"),
            selectmode="extended",
            height=SOURCE_JOBS_TREE_VISIBLE_ROWS,
            style="Jobs.Treeview")
        self.source_jobs_tree.heading(
            "#0",
            text=SOURCE_JOB_HEADINGS["job"])
        self.source_jobs_tree.column(
            "#0",
            width=180,
            minwidth=90,
            stretch=False)
        for column in columns:
            self.source_jobs_tree.heading(
                column,
                text=SOURCE_JOB_HEADINGS[column])
            self.source_jobs_tree.column(
                column,
                width=(
                    SOURCE_JOB_DETAIL_MIN_WIDTH
                    if column == "detail"
                    else 100),
                minwidth=(
                    SOURCE_JOB_DETAIL_MIN_WIDTH
                    if column == "detail"
                    else 60),
                stretch=column == "detail")
        jobs_scrollbar = RoundedScrollbar(
            table,
            command=self.source_jobs_tree.yview,
            background=self.PANEL_BACKGROUND,
            active=self.ACCENT)
        self.source_jobs_tree.configure(
            yscrollcommand=jobs_scrollbar.set)
        self.source_jobs_tree.grid(
            row=0,
            column=0,
            sticky="nsew")
        jobs_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
            padx=(7, 0))
        self.source_jobs_tree.tag_configure(
            "failed",
            foreground=self.ERROR)
        for failed_status in (
                "cancelled",
                "connection_failed",
                "invalid",
                "invalid_response"):
            self.source_jobs_tree.tag_configure(
                failed_status,
                foreground=self.ERROR)
        self.source_jobs_tree.tag_configure(
            "running",
            foreground=self.WARNING)
        self.source_jobs_tree.tag_configure(
            "completed",
            foreground=self.ACCENT)
        self.source_jobs_tree.tag_configure(
            "succeeded",
            foreground=self.ACCENT)
        self.source_jobs_tree.tag_configure(
            "validation_error",
            foreground=self.ERROR,
            font=("DejaVu Sans", 9, "bold"))
        self.source_jobs_tree.tag_configure(
            "job_group",
            background=self.PALE,
            foreground=self.TEXT_PRIMARY,
            font=("DejaVu Sans", 9, "bold"))
        self.source_jobs_tree.tag_configure(
            "job_even",
            background=self.PANEL_BACKGROUND)
        self.source_jobs_tree.tag_configure(
            "job_odd",
            background="#F7F8F6")
        self.source_jobs_tree.bind(
            "<<TreeviewSelect>>",
            self._source_job_selection_changed,
            add="+")
        self.source_jobs_tree.bind(
            "<Configure>",
            self._resize_source_job_columns,
            add="+")
        self.source_job_by_tree_id = {}
        self.source_job_display_rows = ()

        latest_detail = ttk.Frame(table, style="Panel.TFrame")
        latest_detail.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(8, 0))
        latest_detail.columnconfigure(0, weight=1)
        latest_detail.rowconfigure(1, weight=1)
        self.source_latest_detail_heading = tk.StringVar(
            value="FULL LATEST DETAIL")
        self.source_latest_detail = tk.StringVar(
            value=(
                "Select one request chunk or Deck stage above to read its "
                "complete latest detail here."
            ))
        ttk.Label(
            latest_detail,
            textvariable=self.source_latest_detail_heading,
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w")
        self.source_latest_detail_text = tk.Text(
            latest_detail,
            height=3,
            wrap=tk.WORD,
            font=("DejaVu Sans", 9),
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_SECONDARY,
            selectbackground="#D7E9E5",
            selectforeground=self.TEXT_PRIMARY,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=0,
            pady=3,
            state=tk.DISABLED)
        latest_detail_scrollbar = RoundedScrollbar(
            latest_detail,
            command=self.source_latest_detail_text.yview,
            background=self.PANEL_BACKGROUND,
            active=self.ACCENT)
        self.source_latest_detail_text.configure(
            yscrollcommand=latest_detail_scrollbar.set)
        self.source_latest_detail_text.grid(
            row=1,
            column=0,
            sticky="nsew",
            pady=(3, 0))
        latest_detail_scrollbar.grid(
            row=1,
            column=1,
            sticky="ns",
            padx=(6, 0),
            pady=(3, 0))
        self.source_latest_detail_text.configure(state=tk.NORMAL)
        self.source_latest_detail_text.insert(
            "1.0",
            self.source_latest_detail.get())
        self.source_latest_detail_text.configure(state=tk.DISABLED)
        self.source_latest_detail_text.bind(
            "<FocusOut>",
            self._clear_text_selection,
            add="+")

        actions = ttk.Frame(page, style="App.TFrame")
        actions.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(10, 0))
        safety_controls = ttk.Frame(actions, style="App.TFrame")
        safety_controls.grid(row=0, column=0, sticky="ew")
        safety_controls.columnconfigure(2, weight=1)
        self.source_pause_button = ttk.Button(
            safety_controls,
            text="Emergency pause",
            command=self._pause_source_dispatch,
            state=(
                tk.NORMAL
                if self.source_pause_callback is not None
                else tk.DISABLED),
            style="Accent.TButton",
            cursor="hand2")
        self.source_pause_button.grid(
            row=0,
            column=0,
            sticky="w")
        self.source_resume_button = ttk.Button(
            safety_controls,
            text="Resume queued work",
            command=self._resume_source_dispatch,
            state=(
                tk.NORMAL
                if self.source_resume_callback is not None
                else tk.DISABLED),
            style="CompactSecondary.TButton",
            cursor="hand2")
        self.source_resume_button.grid(
            row=0,
            column=1,
            sticky="w",
            padx=(8, 12))
        ttk.Label(
            safety_controls,
            textvariable=self.source_dispatch_status,
            style="Status.TLabel",
            wraplength=760).grid(
                row=0,
                column=2,
                sticky="w")
        ttk.Label(
            safety_controls,
            text=(
                "Pause prevents every queued or otherwise unsent paid OpenAI "
                "request from this app. A request already sent cannot be "
                "recalled and may still finish or incur its charge."
            ),
            style="Status.TLabel",
            wraplength=1050).grid(
                row=1,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(5, 0))
        action_buttons = ttk.Frame(actions, style="App.TFrame")
        action_buttons.grid(
            row=1,
            column=0,
            sticky="w",
            pady=(9, 0))
        ttk.Button(
            action_buttons,
            text="Refresh",
            command=self._refresh_source_jobs,
            style="CompactSecondary.TButton",
            cursor="hand2").grid(
                row=0,
                column=0,
                sticky="w")
        self.source_inspect_button = ttk.Button(
            action_buttons,
            text="Inspect selected stage",
            command=self._inspect_selected_source_job,
            style="CompactSecondary.TButton",
            cursor="hand2")
        self.source_inspect_button.grid(
            row=0,
            column=1,
            sticky="w",
            padx=(8, 0))
        self.source_view_problems_button = ttk.Button(
            action_buttons,
            text="View problems…",
            command=self._open_selected_source_problems,
            state=tk.DISABLED,
            style="CompactSecondary.TButton",
            cursor="hand2")
        self.source_view_problems_button.grid(
            row=0,
            column=2,
            sticky="w",
            padx=(8, 0))
        self.source_retry_selected_button = ttk.Button(
            action_buttons,
            text="Resume / retry selected…",
            command=self._retry_selected_source_jobs,
            state=(
                tk.NORMAL
                if self.source_retry_callback is not None
                else tk.DISABLED),
            style="CompactSecondary.TButton",
            cursor="hand2")
        self.source_retry_selected_button.grid(
            row=0,
            column=3,
            sticky="w",
            padx=(8, 0))
        self.source_retry_all_button = ttk.Button(
            action_buttons,
            text="Resume / retry all incomplete…",
            command=self._retry_all_source_jobs,
            state=(
                tk.NORMAL
                if self.source_retry_callback is not None
                else tk.DISABLED),
            style="CompactSecondary.TButton",
            cursor="hand2")
        self.source_retry_all_button.grid(
            row=0,
            column=4,
            sticky="w",
            padx=(8, 0))
        ttk.Label(
            actions,
            textvariable=self.source_action_status,
            style="Status.TLabel",
            justify=tk.LEFT,
            wraplength=1050).grid(
                row=2,
                column=0,
                sticky="ew",
                pady=(5, 0))

        inspect_surface = RoundedPanel(
            page,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.WINDOW_BACKGROUND,
            radius=22,
            inset=8)
        inspect_surface.grid(
            row=3,
            column=0,
            sticky="nsew",
            pady=(10, 0))
        inspect_panel = inspect_surface.interior
        inspect_panel.columnconfigure(0, weight=1)
        inspect_panel.rowconfigure(2, weight=1)
        ttk.Label(
            inspect_panel,
            text="INSPECT ONE SAVED STAGE",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                padx=7,
                pady=(4, 3))
        self.source_inspection_scope = tk.StringVar(
            value=(
                "Select one request chunk or Deck stage above. The two tabs "
                "will show what went in and what came back."
            ))
        ttk.Label(
            inspect_panel,
            textvariable=self.source_inspection_scope,
            style="Muted.TLabel",
            wraplength=1100).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=7,
            pady=(0, 5))

        self.source_job_inspection_notebook = ttk.Notebook(inspect_panel)
        self.source_job_inspection_notebook.grid(
            row=2,
            column=0,
            sticky="nsew")
        request_tab = ttk.Frame(
            self.source_job_inspection_notebook,
            style="Panel.TFrame")
        response_tab = ttk.Frame(
            self.source_job_inspection_notebook,
            style="Panel.TFrame")
        self.source_job_inspection_notebook.add(
            request_tab,
            text="Request · what went in")
        self.source_job_inspection_notebook.add(
            response_tab,
            text="Response · what came back")
        self.source_job_request_inspection = self._build_source_job_text_tab(
            request_tab,
            (
                "One saved chunk only: its vocabulary/context input and the "
                "OpenAI response rules used for that request."
            ))
        self.source_job_response_inspection = self._build_source_job_text_tab(
            response_tab,
            (
                "Every saved attempt for this chunk. “raw_text” is the model "
                "reply; “error” records why AutoAnki rejected an attempt; "
                "“validated” is an accepted result."
            ))
        # Preserve the old attribute for small integrations that customize the
        # inspection widget. It now refers to the response tab.
        self.source_job_inspection = self.source_job_response_inspection
        viewport.bind_mousewheel_tree(
            preserve_scrollable_children=True)
        self._refresh_source_dispatch_status()

    def _ensure_source_jobs_page_built(self):
        if not getattr(self, "_source_jobs_page_built", False):
            self._build_source_jobs_page()

    def _build_source_job_text_tab(self, tab, explanation):
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        ttk.Label(
            tab,
            text=explanation,
            style="Muted.TLabel",
            wraplength=1050).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=8,
            pady=(7, 2))
        text = tk.Text(
            tab,
            height=SOURCE_JOB_INSPECTION_VISIBLE_LINES,
            wrap=tk.WORD,
            font=("DejaVu Sans Mono", 9),
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            selectbackground="#D7E9E5",
            selectforeground=self.TEXT_PRIMARY,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=8,
            pady=6,
            state=tk.DISABLED)
        scrollbar = RoundedScrollbar(
            tab,
            command=text.yview,
            background=self.PANEL_BACKGROUND,
            active=self.ACCENT)
        text.configure(yscrollcommand=scrollbar.set)
        text.grid(
            row=1,
            column=0,
            sticky="nsew")
        scrollbar.grid(
            row=1,
            column=1,
            sticky="ns")
        text.bind(
            "<FocusOut>",
            self._clear_text_selection,
            add="+")
        text.configure(state=tk.NORMAL)
        text.insert(
            "1.0",
            "Select one stage above, then choose “Inspect selected stage”.")
        text.configure(state=tk.DISABLED)
        return text

    def _set_source_latest_detail(
            self,
            text,
            *,
            validation_error=False):
        self.source_latest_detail.set(text)
        if not hasattr(self, "source_latest_detail_text"):
            return
        self._replace_readonly_text(
            self.source_latest_detail_text,
            text)
        self.source_latest_detail_text.configure(
            foreground=(
                self.ERROR
                if validation_error
                else self.TEXT_SECONDARY))

    def _resize_source_job_columns(self, event=None):
        if not hasattr(self, "source_jobs_tree"):
            return
        available_width = (
            event.width
            if event is not None
            else self.source_jobs_tree.winfo_width())
        style = ttk.Style(self.root)
        body_font = style.lookup(
            "Jobs.Treeview",
            "font") or ("DejaVu Sans", 9)
        heading_font = style.lookup(
            "Jobs.Treeview.Heading",
            "font") or ("DejaVu Sans", 9, "bold")

        def measure(value):
            text = str(value)
            return max(
                int(self.root.tk.call(
                    "font",
                    "measure",
                    body_font,
                    text)),
                int(self.root.tk.call(
                    "font",
                    "measure",
                    heading_font,
                    text)))

        widths = source_job_column_widths(
            self.source_job_display_rows,
            measure,
            max(1, available_width))
        self.source_jobs_tree.column(
            "#0",
            width=widths["job"],
            minwidth=40,
            stretch=False)
        for column in SOURCE_JOB_COLUMNS:
            width = widths[column]
            self.source_jobs_tree.column(
                column,
                width=width,
                minwidth=40,
                stretch=False)

    def _source_preview_request(self, offset=0):
        option = self._selected_source_option()
        page_size = parse_source_chunk_size(
            self.source_preview_page_size.get())
        if (
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset < 0):
            raise ValueError(
                "Source preview offset must be a non-negative integer.")
        return {
            "source_key": option.key,
            "offset": offset,
            "limit": page_size,
        }

    def _source_preview_selection_changed(self, _event=None):
        self._source_selection_changed()
        self._clear_source_preview(
            "Loading the selected source’s local metadata…")
        self._request_source_preview(offset=0)

    def _reload_source_preview_from_start(self, _event=None):
        self._request_source_preview(offset=0)

    def _request_source_preview(self, *, offset):
        if self.source_preview_loader is None:
            self.source_preview_status.set(
                "Source inspection is unavailable in this build.")
            self._update_source_preview_navigation()
            return
        try:
            request = self._source_preview_request(offset)
        except ValueError as error:
            self.source_preview_status.set(str(error))
            self._update_source_preview_navigation()
            return

        self.source_preview_generation += 1
        generation = self.source_preview_generation
        self.source_preview_pending.add(generation)
        self.source_preview_status.set(
            "Loading local vocabulary metadata…")
        self._update_source_preview_navigation()

        def load_in_background():
            try:
                value = self.source_preview_loader(request)
            except Exception as error:
                self.source_preview_result_queue.put((
                    generation,
                    request,
                    "error",
                    error,
                ))
            else:
                self.source_preview_result_queue.put((
                    generation,
                    request,
                    "success",
                    value,
                ))

        threading.Thread(
            target=load_in_background,
            daemon=True).start()
        if not self.source_preview_polling:
            self.source_preview_polling = True
            self.root.after(
                self.POLL_INTERVAL_MS,
                self._poll_source_preview)

    def _poll_source_preview(self):
        try:
            generation, request, outcome, value = (
                self.source_preview_result_queue.get_nowait())
        except queue.Empty:
            if self.source_preview_pending:
                self.root.after(
                    self.POLL_INTERVAL_MS,
                    self._poll_source_preview)
            else:
                self.source_preview_polling = False
                self._update_source_preview_navigation()
            return

        self.source_preview_pending.discard(generation)
        if generation == self.source_preview_generation:
            if outcome == "error":
                self.source_preview_status.set(
                    f"Could not load source preview · {value}")
            else:
                try:
                    page = normalise_source_preview_response(value)
                    if page.source_key != request["source_key"]:
                        raise ValueError(
                            "Source preview returned data for another source.")
                    if page.offset != request["offset"]:
                        raise ValueError(
                            "Source preview returned a different page offset.")
                    selected_key = self._selected_source_option().key
                    if selected_key != request["source_key"]:
                        raise ValueError(
                            "The selected source changed while loading.")
                except (TypeError, ValueError) as error:
                    self.source_preview_status.set(
                        f"Invalid source preview · {error}")
                else:
                    self._apply_source_preview(page)

        if self.source_preview_pending:
            self.root.after(
                self.POLL_INTERVAL_MS,
                self._poll_source_preview)
        else:
            self.source_preview_polling = False
            self._update_source_preview_navigation()

    def _apply_source_preview(self, page):
        for item_id in self.source_preview_tree.get_children():
            self.source_preview_tree.delete(item_id)
        self.source_preview_by_tree_id = {}
        self.source_preview_page_data = page
        for index, item in enumerate(page.items):
            item_id = f"source_preview_{index}"
            self.source_preview_tree.insert(
                "",
                tk.END,
                iid=item_id,
                values=(
                    f"{item.rank:,}",
                    item.term,
                    item.section_title or "—",
                ))
            self.source_preview_by_tree_id[item_id] = item

        if page.items:
            first = page.offset + 1
            last = page.offset + len(page.items)
            self.source_preview_status.set(
                f"Showing {first:,}–{last:,} of {page.total:,} candidates · "
                "local metadata only")
            first_id = "source_preview_0"
            self.source_preview_tree.selection_set(first_id)
            self.source_preview_tree.focus(first_id)
            self.source_preview_tree.see(first_id)
            self._show_source_preview_item(page.items[0])
        else:
            self.source_preview_status.set(
                "This prepared source has no vocabulary candidates.")
            self.source_preview_detail_title.set(
                "NO VOCABULARY CANDIDATES")
            self._replace_source_preview_detail(
                (("heading", "Source metadata"),),
                (("context", "Nothing is available to inspect."),))
        self._update_source_preview_navigation()

    def _clear_source_preview(self, status=None):
        if hasattr(self, "source_preview_tree"):
            for item_id in self.source_preview_tree.get_children():
                self.source_preview_tree.delete(item_id)
        self.source_preview_page_data = None
        self.source_preview_by_tree_id = {}
        if hasattr(self, "source_preview_detail_title"):
            self.source_preview_detail_title.set(
                "SELECT A WORD TO INSPECT ITS RETAINED CONTEXT")
        if hasattr(self, "source_preview_detail"):
            self._replace_source_preview_detail(
                (("heading", "Source metadata"),),
                ((
                    "context",
                    "Load a page and select an ordered vocabulary candidate.",
                ),))
        if status is not None:
            self.source_preview_status.set(status)
        self._update_source_preview_navigation()

    def _source_preview_item_selected(self, _event=None):
        selection = self.source_preview_tree.selection()
        if not selection:
            return
        item = self.source_preview_by_tree_id.get(selection[0])
        if item is not None:
            self._show_source_preview_item(item)

    def _show_source_preview_item(self, item):
        self.source_preview_detail_title.set(
            f"#{item.rank:,}  ·  {item.term}")
        section_title = item.section_title or "Not recorded"
        previous = item.previous_sentence or "Not available"
        current = item.current_sentence or "Not available"
        following = item.next_sentence or "Not available"
        self._replace_source_preview_detail(
            (
                ("heading", "CHAPTER / SECTION"),
                ("context", section_title),
                ("heading", "PREVIOUS SENTENCE"),
                ("context", previous),
                ("heading", "CURRENT SENTENCE"),
                ("current", current),
                ("heading", "NEXT SENTENCE"),
                ("context", following),
            ))

    def _replace_source_preview_detail(self, *groups):
        self.source_preview_detail.configure(state=tk.NORMAL)
        self.source_preview_detail.delete("1.0", tk.END)
        for group in groups:
            for tag, text in group:
                self.source_preview_detail.insert(
                    tk.END,
                    f"{text}\n",
                    tag)
        self.source_preview_detail.configure(state=tk.DISABLED)
        self.source_preview_detail.yview_moveto(0)

    def _source_preview_previous_page(self):
        page = self.source_preview_page_data
        if page is None:
            return
        try:
            page_size = parse_source_chunk_size(
                self.source_preview_page_size.get())
        except ValueError as error:
            self.source_preview_status.set(str(error))
            return
        self._request_source_preview(
            offset=max(0, page.offset - page_size))

    def _source_preview_next_page(self):
        page = self.source_preview_page_data
        if page is None:
            return
        next_offset = page.offset + len(page.items)
        if next_offset < page.total:
            self._request_source_preview(offset=next_offset)

    def _update_source_preview_navigation(self):
        if not hasattr(self, "source_preview_previous_button"):
            return
        busy = (
            self.source_preview_generation
            in self.source_preview_pending)
        available = self.source_preview_loader is not None
        self.source_preview_load_button.configure(
            state=tk.NORMAL if available and not busy else tk.DISABLED)
        page = self.source_preview_page_data
        previous_enabled = (
            available
            and not busy
            and page is not None
            and page.offset > 0)
        next_enabled = (
            available
            and not busy
            and page is not None
            and page.offset + len(page.items) < page.total)
        self.source_preview_previous_button.configure(
            state=tk.NORMAL if previous_enabled else tk.DISABLED)
        self.source_preview_next_button.configure(
            state=tk.NORMAL if next_enabled else tk.DISABLED)

    def _load_source_catalogue(self):
        selected_option = self.source_options_by_label.get(
            self.source_selected_label.get())
        selected_key = (
            selected_option.key
            if selected_option is not None
            else getattr(
                self,
                "gui_preferences",
                {}).get("source_key") or None)
        loaded = ()
        error = None
        if self.source_catalog_loader is not None:
            try:
                loaded = self.source_catalog_loader()
                if isinstance(loaded, dict):
                    loaded = loaded.get("sources", ())
            except Exception as caught_error:
                error = caught_error
                loaded = ()
        merged = {
            option.key: option
            for option in BUILT_IN_SOURCE_OPTIONS
        }
        merged.update({
            option.key: option
            for option in normalise_source_options(loaded)
        })
        self.source_options = tuple(merged.values())
        self.source_options_by_key = {
            option.key: option
            for option in self.source_options
        }
        self.source_display_labels_by_key = source_option_display_labels(
            self.source_options)
        self.source_options_by_label = {
            self.source_display_labels_by_key[option.key]: option
            for option in self.source_options
        }
        labels = tuple(
            self.source_display_labels_by_key[option.key]
            for option in self.source_options)
        self.source_selector.configure(values=labels)
        self.source_preview_selector.configure(values=labels)
        if selected_key in self.source_display_labels_by_key:
            self.source_selected_label.set(
                self.source_display_labels_by_key[selected_key])
        elif (
                self.source_selected_label.get()
                not in self.source_options_by_label):
            self.source_selected_label.set(
                labels[0] if labels else "")
        if error is not None:
            self.source_catalog_status.set(
                f"Could not refresh processed sources · {error}")
        else:
            self.source_catalog_status.set(
                f"{len(labels)} prepared sources available")
        self._source_selection_changed()

    def _refresh_source_catalogue(self):
        self._load_source_catalogue()
        self.source_action_status.set(
            self.source_catalog_status.get())

    def _selected_source_option(self):
        option = self.source_options_by_label.get(
            self.source_selected_label.get())
        if option is None:
            raise ValueError("Select a prepared source.")
        return option

    def _source_selection_changed(self, _event=None):
        option = None
        try:
            option = self._selected_source_option()
        except ValueError:
            self.source_summary.set("No prepared source selected.")
            self.source_deck_notice.set(
                "Select a source to choose the output deck.")
        else:
            if option.source_language_key:
                preferred_language_key = getattr(
                    self,
                    "gui_preferences",
                    {}).get(
                    "source_language_overrides",
                    {}).get(
                        option.key,
                        option.source_language_key)
                try:
                    source_language = pipeline_store.get_language(
                        preferred_language_key)
                except ValueError:
                    try:
                        source_language = pipeline_store.get_language(
                            option.source_language_key)
                    except ValueError:
                        self.source_language_label.set("")
                    else:
                        self.source_language_label.set(source_language.name)
                else:
                    self.source_language_label.set(source_language.name)
            summary = []
            if option.word_count is not None:
                summary.append(
                    f"{option.word_count:,} unique candidate words")
            if option.token_occurrence_count is not None:
                summary.append(
                    f"{option.token_occurrence_count:,} running-word "
                    "occurrences")
            if option.section_count is not None:
                summary.append(
                    f"{option.section_count:,} source sections")
            summary.append(
                "built-in preset"
                if option.preset
                else "locally prepared source")
            self.source_summary.set(" · ".join(summary))
            self.source_deck_notice.set(
                (
                    "Creates and imports separate card-type subdecks under "
                    f'“Vocabulary from {option.name}”.'
                    if (
                        hasattr(self, "source_separate_decks")
                        and self.source_separate_decks.get())
                    else (
                        f'Creates and imports “Vocabulary from '
                        f'{option.name}”.')))
        if (
                self.source_preview_page_data is not None
                and (
                    option is None
                    or self.source_preview_page_data.source_key
                    != option.key)):
            self._clear_source_preview(
                "The source changed. Load a page to inspect it.")
        self.source_paid_authorized.set(False)
        self.source_zero_notice_shown = False
        self._refresh_card_setup_generation_context()
        self._sync_learned_filter_controls()
        self._schedule_source_estimate()

    def _source_language_changed(self, _event=None):
        self.source_paid_authorized.set(False)
        self.source_zero_notice_shown = False
        self._refresh_card_setup_generation_context()
        self._sync_learned_filter_controls()
        self._schedule_source_estimate()

    def _source_context_changed(self, _event=None):
        if (
                getattr(
                    self,
                    "source_use_source_examples",
                    None) is not None
                and self.source_use_source_examples.get()
                and self.source_context_label.get()
                != SOURCE_CONTEXT_LABELS["sentence"]):
            self.source_context_label.set(
                SOURCE_CONTEXT_LABELS["sentence"])
        self._update_source_context_description()
        self.source_paid_authorized.set(False)
        self._schedule_source_estimate()

    def _source_protocol_changed(self, _event=None):
        protocol = SOURCE_PROTOCOL_KEYS_BY_LABEL.get(
            self.source_request_protocol_label.get())
        if protocol == "v8":
            current_reasoning = SOURCE_REASONING_KEYS_BY_LABEL.get(
                self.source_reasoning_label.get(),
                "low")
            if current_reasoning != "low":
                self._source_reasoning_before_v8 = current_reasoning
            self.source_reasoning_label.set(
                dict(SOURCE_REASONING_OPTIONS)["low"])
            if hasattr(self, "source_reasoning_selector"):
                self.source_reasoning_selector.configure(
                    state=tk.DISABLED)
        else:
            remembered_reasoning = getattr(
                self,
                "_source_reasoning_before_v8",
                "low")
            if remembered_reasoning in dict(SOURCE_REASONING_OPTIONS):
                self.source_reasoning_label.set(
                    dict(SOURCE_REASONING_OPTIONS)[remembered_reasoning])
            if hasattr(self, "source_reasoning_selector"):
                self.source_reasoning_selector.configure(
                    state="readonly")
        self.source_paid_authorized.set(False)
        self._schedule_source_estimate()

    def _source_reasoning_changed(self, _event=None):
        self.source_paid_authorized.set(False)
        self._schedule_source_estimate()

    def _selected_source_model_key(self):
        variable = getattr(self, "source_model_label", None)
        label = (
            variable.get()
            if hasattr(variable, "get")
            else SOURCE_MODEL_OPTIONS[0][1])
        return SOURCE_MODEL_KEYS_BY_LABEL.get(
            label,
            SOURCE_MODEL_OPTIONS[0][0])

    def _source_model_changed(self, _event=None):
        # Model-specific constraints are applied to the effective request.
        # Keep the user's cloud/local tuning intact while a control is
        # temporarily unavailable so switching back restores it.
        self._sync_source_model_controls()
        self.source_paid_authorized.set(False)
        self._schedule_source_estimate()

    def _sync_source_model_controls(self):
        model = self._selected_source_model_key()
        profile = source_model_profile(model)
        local = profile.local
        if hasattr(self, "source_protocol_selector"):
            self.source_protocol_selector.configure(
                state=tk.DISABLED if local else "readonly")
        if hasattr(self, "source_execution_selector"):
            self.source_execution_selector.configure(
                state=tk.DISABLED if local else "readonly")
        if hasattr(self, "source_web_search_check"):
            self.source_web_search_check.configure(
                state=tk.DISABLED if local else tk.NORMAL)
        if hasattr(self, "source_automatic_repair_check"):
            self.source_automatic_repair_check.configure(
                text=(
                    "Automatic repair (exact translation reuse + up to 5 "
                    "local validation follow-ups per failed request)"
                    if local
                    else (
                        "Automatic repair (exact translation reuse + up to "
                        "3 additional paid micro-repair calls per failed "
                        "request)")))
        if hasattr(self, "source_authorization_check"):
            self.source_authorization_check.configure(
                text=(
                    "I authorise the paid OpenAI requests shown in this "
                    "estimate"))
        estimate_heading = getattr(
            self,
            "source_estimate_heading_label",
            None)
        if estimate_heading is not None:
            estimate_heading.configure(
                text=(
                    "LOCAL MODEL COST"
                    if local
                    else "ESTIMATED OPENAI COST"))

    def _source_automatic_repair_changed(self):
        automatic_variable = getattr(
            self,
            "source_automatic_repair",
            None)
        enabled = bool(
            automatic_variable.get()
            if hasattr(automatic_variable, "get")
            else False)
        mode = SOURCE_EXECUTION_KEYS_BY_LABEL.get(
            self.source_execution_label.get())
        if enabled and mode == "economy":
            self.source_execution_label.set(
                dict(SOURCE_EXECUTION_OPTIONS)["standard"])
            self._source_execution_mode_changed()
            return
        self.source_paid_authorized.set(False)
        self._schedule_source_estimate()

    def _source_execution_mode_changed(self, _event=None):
        mode = SOURCE_EXECUTION_KEYS_BY_LABEL.get(
            self.source_execution_label.get())
        local = is_local_source_model(
            self._selected_source_model_key())
        economy = mode == "economy" and not local
        if hasattr(self, "source_automatic_repair_check"):
            self.source_automatic_repair_check.configure(
                state=tk.DISABLED if economy else tk.NORMAL)
        entry_state = (
            tk.DISABLED
            if economy or local
            else tk.NORMAL)
        if hasattr(self, "source_concurrency_entry"):
            self.source_concurrency_entry.configure(state=entry_state)
        if hasattr(self, "source_stagger_entry"):
            self.source_stagger_entry.configure(state=entry_state)
        if hasattr(self, "source_rate_notice_label"):
            self.source_rate_notice_label.configure(
                text=(
                    "Economy sends one asynchronous Batch file. Parallel "
                    "workers, staggering, and automatic live-request retries "
                    "do not apply; resume the saved job later to collect its "
                    "results without resubmitting completed work."
                    if economy
                    else (
                        (
                            "Local inference is sequential to protect GPU "
                            "and system memory. Invalid responses can use up "
                            "to five persisted validation follow-ups without "
                            "an API charge."
                        )
                        if local
                        else
                        "The default starts eight workers, with requests "
                        "launched no faster than the stagger permits. Account "
                        "limits vary; the coordinator observes OpenAI "
                        "rate-limit responses and backs off automatically. "
                        "Connection/time-out, HTTP 429, and HTTP 5xx failures "
                        "may retry without asking; retry costs are not "
                        "included in the estimate."
                    )))
        if hasattr(self, "source_generate_button"):
            self.source_generate_button.configure(
                text=(
                    "Submit Economy Batch"
                    if economy
                    else (
                        "Generate locally and import source deck"
                        if local
                        else "Generate and import source deck")))
        self._sync_source_model_controls()
        self.source_paid_authorized.set(False)
        self._schedule_source_estimate()

    def _source_example_setting_changed(self):
        if self.source_use_source_examples.get():
            current_context = SOURCE_CONTEXT_KEYS_BY_LABEL.get(
                self.source_context_label.get(),
                "sentence")
            if current_context != "sentence":
                self._source_context_before_source_examples = current_context
            self.source_context_label.set(
                SOURCE_CONTEXT_LABELS["sentence"])
            self._update_source_context_description()
        else:
            remembered_context = getattr(
                self,
                "_source_context_before_source_examples",
                "sentence")
            if remembered_context in SOURCE_CONTEXT_LABELS:
                self.source_context_label.set(
                    SOURCE_CONTEXT_LABELS[remembered_context])
                self._update_source_context_description()
        self._sync_source_card_options()
        self.source_paid_authorized.set(False)
        self._schedule_source_estimate()

    def _selected_source_direction_keys(self):
        variables = getattr(
            self,
            "source_card_direction_variables",
            {})
        return tuple(
            direction.key
            for direction in pipeline_store.list_directions()
            if (
                direction.key in variables
                and variables[direction.key].get()))

    def _sync_source_card_options(self):
        selected = self._selected_source_direction_keys()
        names = {
            "context": "Sentence → Meaning",
            "word_to_meaning": "Word → Meaning",
            "meaning_to_word": "Meaning → Word",
        }
        summary = getattr(
            self,
            "source_card_direction_summary",
            None)
        if hasattr(summary, "set"):
            summary.set(
                ", ".join(names[key] for key in selected)
                if selected
                else "Select at least one card type")
        use_source = bool(
            getattr(
                self,
                "source_use_source_examples",
                None).get()
            if hasattr(
                getattr(self, "source_use_source_examples", None),
                "get")
            else False)
        nuance_enabled = (
            use_source
            and "context" in selected)
        context_selector = getattr(
            self,
            "source_context_selector",
            None)
        if context_selector is not None:
            context_selector.configure(
                state=tk.DISABLED if use_source else "readonly")
        nuance_check = getattr(
            self,
            "source_include_context_nuance_check",
            None)
        if nuance_check is not None:
            nuance_check.configure(
                state=tk.NORMAL if nuance_enabled else tk.DISABLED)

    def _source_card_options_changed(self):
        self._sync_source_card_options()
        self._sync_source_example_control()
        try:
            option = self._selected_source_option()
        except ValueError:
            pass
        else:
            self.source_deck_notice.set(
                (
                    "Creates and imports separate card-type subdecks under "
                    f'“Vocabulary from {option.name}”.'
                    if self.source_separate_decks.get()
                    else (
                        f'Creates and imports “Vocabulary from '
                        f'{option.name}”.')))
        self.source_paid_authorized.set(False)
        self._schedule_source_estimate()

    def _source_prefix_setting_changed(self):
        """Apply the prefix toggle and invalidate the paid estimate."""
        self._sync_source_prefix_control()
        self.source_paid_authorized.set(False)
        self.source_zero_notice_shown = False
        self._schedule_source_estimate()

    def _sync_source_prefix_control(self):
        """Enable prefix length entry only when the prefix option is active."""
        variable = getattr(self, "source_limit_to_prefix", None)
        entry = getattr(self, "source_prefix_token_entry", None)
        if entry is None:
            return
        enabled = bool(
            variable.get()
            if variable is not None and hasattr(variable, "get")
            else False)
        entry.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def _update_source_context_description(self):
        key = SOURCE_CONTEXT_KEYS_BY_LABEL.get(
            self.source_context_label.get(),
            "sentence")
        self.source_context_description.set(
            SOURCE_CONTEXT_DESCRIPTIONS[key])
        self._sync_source_example_control()

    def _sync_source_example_control(self):
        source_example_variable = getattr(
            self,
            "source_use_source_examples",
            None)
        source_example_check = getattr(
            self,
            "source_use_source_examples_check",
            None)
        context_key = SOURCE_CONTEXT_KEYS_BY_LABEL.get(
            self.source_context_label.get(),
            "sentence")
        selected_directions = self._selected_source_direction_keys()
        enabled = (
            context_key != "none"
            and "context" in selected_directions)
        if not enabled:
            if source_example_check is not None:
                source_example_check.configure(state=tk.DISABLED)
        elif source_example_check is not None:
            source_example_check.configure(state=tk.NORMAL)
        self._sync_source_card_options()

    def _manual_filter_changed(self):
        language_key = self.get_generation_language().key
        self._set_language_filter_enabled(
            language_key,
            self.manual_filter_enabled.get())
        self._update_manual_filter_summary()

    def _manual_filter_selection_changed(self, *_args):
        self._update_manual_filter_summary()

    def _update_manual_filter_summary(self):
        settings = self._language_filter(
            self.get_generation_language().key)
        if not settings.enabled:
            summary = "Learned-word filter is off"
        else:
            complete = sum(source.complete for source in settings.sources)
            summary = (
                f"{complete} learned field"
                f"{'' if complete == 1 else 's'} configured")
            if complete != len(settings.sources) or not complete:
                summary += " · finish the incomplete row"
        self.manual_filter_summary.set(summary)

    def _open_manual_filter_dialog(self):
        self._open_learned_filter_dialog(
            self.get_generation_language().key)

    def _open_source_filter_dialog(self):
        option = self._selected_source_option()
        self._open_learned_filter_dialog(option.source_language_key)

    def _language_filter(self, language_key):
        return learned_filter_store.get_language_filter(
            self.learned_filter_settings,
            language_key)

    def _save_language_filter(self, settings):
        self.learned_filter_settings = (
            learned_filter_store.replace_language_filter(
                self.learned_filter_settings,
                settings))
        try:
            self.learned_filter_saver(self.learned_filter_settings)
        except OSError as error:
            messagebox.showerror(
                "Could not save learned-word settings",
                str(error),
                parent=self.root)
        self._sync_learned_filter_controls()
        for editor in getattr(self, "pipeline_rows", ()):
            refresh = getattr(
                editor,
                "refresh_learned_filter_controls",
                None)
            if refresh is not None:
                refresh()
        if hasattr(self, "source_paid_authorized"):
            self.source_paid_authorized.set(False)
        if hasattr(self, "source_estimate_price"):
            self._schedule_source_estimate()

    def _set_language_filter_enabled(self, language_key, enabled):
        settings = self._language_filter(language_key)
        self._save_language_filter(
            learned_filter_store.LanguageLearnedFilter(
                language_key=settings.language_key,
                enabled=bool(enabled),
                sources=settings.sources))

    def _sync_learned_filter_controls(self):
        try:
            manual = self._language_filter(
                self.get_generation_language().key)
            self.manual_filter_enabled.set(manual.enabled)
            self._update_manual_filter_summary()
        except (AttributeError, ValueError):
            pass
        try:
            option = self._selected_source_option()
            source = self._language_filter(option.source_language_key)
            self.source_exclude_enabled.set(source.enabled)
            complete = sum(item.complete for item in source.sources)
            if not source.enabled:
                status = "Learned-word filter is off for this language."
            elif complete:
                status = (
                    f"{complete} learned field"
                    f"{'' if complete == 1 else 's'} configured for "
                    f"{pipeline_store.get_language(option.source_language_key).name}.")
            else:
                status = (
                    "Filtering is enabled, but no complete learned field is "
                    "configured.")
            self.source_exclusion_status.set(status)
        except (AttributeError, ValueError):
            pass

    def _open_learned_filter_dialog(self, language_key):
        language_key = learned_filter_store.settings_language_key(
            language_key)
        self.refresh_anki_decks(launch_if_needed=True)
        existing = getattr(self, "manual_filter_dialog", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    self._close_learned_filter_dialog()
            except tk.TclError:
                pass

        dialog = tk.Toplevel(self.root)
        self.manual_filter_dialog = dialog
        self.learned_filter_dialog = dialog
        self.learned_filter_dialog_language_key = language_key
        language_name = pipeline_store.get_language(language_key).name
        dialog.title(f"Learned-word filter · {language_name}")
        dialog.configure(background=self.WINDOW_BACKGROUND)
        dialog.transient(self.root)
        dialog.resizable(True, True)
        dialog.minsize(1100, 650)
        dialog.protocol(
            "WM_DELETE_WINDOW",
            self._close_learned_filter_dialog)

        outer = ttk.Frame(
            dialog,
            padding=(22, 20),
            style="App.TFrame")
        outer.pack(fill="both", expand=True)
        surface = RoundedPanel(
            outer,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.WINDOW_BACKGROUND,
            radius=24,
            inset=18)
        surface.pack(fill="both", expand=True)
        panel = surface.interior
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(3, weight=1)

        ttk.Label(
            panel,
            text=f"OMIT {language_name.upper()} WORDS ALREADY LEARNED",
            style="Section.TLabel").grid(
                row=0,
                column=0,
                sticky="w")
        ttk.Label(
            panel,
            text=(
                "Each row contributes one Anki field. The union of every row "
                "is checked before any paid request. A note type owns its "
                "fields; card templates are deliberately irrelevant here."
            ),
            style="Muted.TLabel",
            wraplength=1080).grid(
                row=1,
                column=0,
                sticky="w",
                pady=(5, 12))
        settings = self._language_filter(language_key)
        enabled_variable = tk.BooleanVar(value=settings.enabled)
        ttk.Checkbutton(
            panel,
            text="Enable the learned-word filter",
            variable=enabled_variable,
            command=lambda: self._set_language_filter_enabled(
                language_key,
                enabled_variable.get()),
            style="Panel.TCheckbutton").grid(
                row=2,
                column=0,
                sticky="w")

        rows_view = ScrollableFrame(
            panel,
            background=self.PANEL_BACKGROUND,
            frame_style="Panel.TFrame")
        rows_view.grid(
            row=3,
            column=0,
            sticky="nsew",
            pady=(12, 0))
        self.learned_filter_rows_container = rows_view.content
        self.learned_filter_rows_container.columnconfigure(0, weight=1)
        self.learned_filter_rows = []
        self.learned_filter_rows_by_id = {}
        sources = settings.sources or (
            learned_filter_store.LearnedWordSource(),)
        for source in sources:
            self._add_learned_filter_row(source=source, save=False)
        rows_view.bind_mousewheel_tree()

        footer = ttk.Frame(panel, style="Panel.TFrame")
        footer.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        footer.columnconfigure(1, weight=1)
        ttk.Button(
            footer,
            text="+ Add another learned field",
            command=self._add_learned_filter_row,
            style="Secondary.TButton",
            cursor="hand2").grid(row=0, column=0, sticky="w")
        ttk.Button(
            footer,
            text="Done",
            command=self._close_learned_filter_dialog,
            style="Accent.TButton",
            cursor="hand2").grid(row=0, column=2, sticky="e")
        dialog.update_idletasks()
        width = max(dialog.winfo_reqwidth(), 1240)
        height = max(dialog.winfo_reqheight(), 760)
        x_position = max(
            self.root.winfo_rootx()
            + (self.root.winfo_width() - width) // 2,
            0)
        y_position = max(
            self.root.winfo_rooty()
            + (self.root.winfo_height() - height) // 2,
            0)
        dialog.geometry(
            f"{width}x{height}+{x_position}+{y_position}")
        dialog.grab_set()

    def _close_learned_filter_dialog(self):
        dialog = getattr(self, "learned_filter_dialog", None)
        self.learned_filter_rows_by_id = {}
        self.learned_filter_rows = []
        self.learned_filter_dialog_language_key = None
        self.learned_filter_dialog = None
        self.manual_filter_dialog = None
        if dialog is not None:
            try:
                if dialog.winfo_exists():
                    dialog.destroy()
            except tk.TclError:
                pass

    def _add_learned_filter_row(self, source=None, *, save=True):
        if source is None:
            source = learned_filter_store.source_for_new_row(
                tuple(
                    self._learned_filter_source_from_row(row)
                    for row in self.learned_filter_rows))
        else:
            source = learned_filter_store.LearnedWordSource.from_mapping(
                source)
        row_id = f"learned_{id(source)}_{len(self.learned_filter_rows)}"
        surface = RoundedPanel(
            self.learned_filter_rows_container,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.PANEL_BACKGROUND,
            radius=18,
            inset=14)
        surface.grid(
            row=len(self.learned_filter_rows),
            column=0,
            sticky="ew",
            pady=(0, 12))
        frame = surface.interior
        frame.columnconfigure(0, weight=1)
        deck_var = tk.StringVar(value=source.deck_name)
        note_var = tk.StringVar(value=source.note_type)
        field_var = tk.StringVar(value=source.field_name)
        title_var = tk.StringVar(
            value=f"LEARNED SOURCE {len(self.learned_filter_rows) + 1}")
        ttk.Label(
            frame,
            textvariable=title_var,
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                pady=(0, 10))
        remove_button = ttk.Button(
            frame,
            text="Remove",
            style="CompactSecondary.TButton",
            cursor="hand2")
        remove_button.grid(
            row=0,
            column=1,
            sticky="e",
            pady=(0, 10))
        for row_number, text in (
                (1, "ANKI DECK"),
                (3, "NOTE TYPE"),
                (5, "FIELD")):
            ttk.Label(
                frame,
                text=text,
                style="FieldLabel.TLabel").grid(
                    row=row_number,
                    column=0,
                    columnspan=2,
                    sticky="w",
                    pady=(0, 4))
        deck_box = ttk.Combobox(
            frame,
            textvariable=deck_var,
            values=self.deck_options,
            state="readonly",
            style="App.TCombobox")
        note_box = ttk.Combobox(
            frame,
            textvariable=note_var,
            values=(() if not source.note_type else (source.note_type,)),
            state=("readonly" if source.deck_name else "disabled"),
            style="App.TCombobox")
        field_box = ttk.Combobox(
            frame,
            textvariable=field_var,
            values=(() if not source.field_name else (source.field_name,)),
            state=("readonly" if source.note_type else "disabled"),
            style="App.TCombobox")
        for row_number, box in (
                (2, deck_box),
                (4, note_box),
                (6, field_box)):
            box.grid(
                row=row_number,
                column=0,
                columnspan=2,
                sticky="ew",
                pady=(0, 11 if row_number < 6 else 0))
            box.bind(
                "<FocusOut>",
                self._clear_entry_selection,
                add="+")
        row = {
            "id": row_id,
            "frame": surface,
            "surface": surface,
            "deck_var": deck_var,
            "note_var": note_var,
            "field_var": field_var,
            "title_var": title_var,
            "deck_box": deck_box,
            "note_box": note_box,
            "field_box": field_box,
        }
        self.learned_filter_rows.append(row)
        self.learned_filter_rows_by_id[row_id] = row
        remove_button.configure(
            command=lambda selected=row: (
                self._remove_learned_filter_row(selected)))
        deck_box.bind(
            "<<ComboboxSelected>>",
            lambda _event, selected=row: (
                self._learned_filter_deck_selected(selected)),
            add="+")
        note_box.bind(
            "<<ComboboxSelected>>",
            lambda _event, selected=row: (
                self._learned_filter_note_selected(selected)),
            add="+")
        field_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._save_learned_filter_rows(),
            add="+")
        if source.deck_name:
            self._request_learned_filter_options(
                row,
                "note_types",
                source.deck_name)
        if save:
            self._save_learned_filter_rows()
        return row

    @staticmethod
    def _learned_filter_source_from_row(row):
        return learned_filter_store.LearnedWordSource(
            deck_name=row["deck_var"].get().strip(),
            note_type=row["note_var"].get().strip(),
            field_name=row["field_var"].get().strip())

    def _save_learned_filter_rows(self):
        if self.learned_filter_dialog_language_key is None:
            return
        existing = self._language_filter(
            self.learned_filter_dialog_language_key)
        self._save_language_filter(
            learned_filter_store.LanguageLearnedFilter(
                language_key=existing.language_key,
                enabled=existing.enabled,
                sources=tuple(
                    self._learned_filter_source_from_row(row)
                    for row in self.learned_filter_rows)))

    def _remove_learned_filter_row(self, row):
        row["frame"].destroy()
        self.learned_filter_rows.remove(row)
        self.learned_filter_rows_by_id.pop(row["id"], None)
        for index, remaining in enumerate(self.learned_filter_rows):
            remaining["frame"].grid_configure(row=index)
            remaining["title_var"].set(
                f"LEARNED SOURCE {index + 1}")
        self._save_learned_filter_rows()

    def _learned_filter_deck_selected(self, row):
        row["note_var"].set("")
        row["field_var"].set("")
        row["note_box"].configure(values=(), state="disabled")
        row["field_box"].configure(values=(), state="disabled")
        self._save_learned_filter_rows()
        deck = row["deck_var"].get().strip()
        if deck:
            self._request_learned_filter_options(
                row,
                "note_types",
                deck)

    def _learned_filter_note_selected(self, row):
        row["field_var"].set("")
        row["field_box"].configure(values=(), state="disabled")
        self._save_learned_filter_rows()
        note_type = row["note_var"].get().strip()
        if note_type:
            self._request_learned_filter_options(
                row,
                "note_type_details",
                note_type)

    def _request_learned_filter_options(self, row, action, value):
        if self.source_anki_options_loader is None:
            self.source_exclusion_status.set(
                "Anki option discovery is unavailable.")
            return
        request = (
            {"action": "note_types", "deck": value}
            if action == "note_types"
            else {
                "action": "note_type_details",
                "note_type": value,
            })
        self.source_anki_options_pending += 1

        def load_in_background():
            try:
                result = self.source_anki_options_loader(request)
            except Exception as error:
                outcome = ("error", error)
            else:
                outcome = ("success", result)
            self.source_anki_options_queue.put((
                row["id"],
                action,
                value,
                *outcome,
            ))

        threading.Thread(
            target=load_in_background,
            daemon=True).start()
        if not self.source_anki_options_polling:
            self.source_anki_options_polling = True
            self.root.after(
                self.POLL_INTERVAL_MS,
                self._poll_source_anki_options)

    def _source_exclusion_changed(self):
        option = self._selected_source_option()
        self._set_language_filter_enabled(
            option.source_language_key,
            self.source_exclude_enabled.get())
        self.source_zero_notice_shown = False

    def _update_source_exclusion_controls(self):
        self._sync_learned_filter_controls()

    def _poll_source_anki_options(self):
        try:
            row_id, action, requested_value, outcome, value = (
                self.source_anki_options_queue.get_nowait())
        except queue.Empty:
            if self.source_anki_options_pending:
                self.root.after(
                    self.POLL_INTERVAL_MS,
                    self._poll_source_anki_options)
            else:
                self.source_anki_options_polling = False
            return
        self.source_anki_options_pending = max(
            0,
            self.source_anki_options_pending - 1)
        row = self.learned_filter_rows_by_id.get(row_id)
        current_value = (
            row["deck_var"].get().strip()
            if row is not None and action == "note_types"
            else (
                row["note_var"].get().strip()
                if row is not None
                else None))
        if row is not None and current_value == requested_value:
            if outcome == "error":
                self.source_exclusion_status.set(
                    f"Could not load Anki options · {value}")
            else:
                try:
                    self._apply_learned_filter_options(
                        row,
                        action,
                        value)
                except (TypeError, ValueError) as error:
                    self.source_exclusion_status.set(
                        f"Anki returned invalid options · {error}")
        if self.source_anki_options_pending:
            self.root.after(
                self.POLL_INTERVAL_MS,
                self._poll_source_anki_options)
        else:
            self.source_anki_options_polling = False

    def _apply_learned_filter_options(self, row, action, value):
        if not isinstance(value, dict):
            raise TypeError("Anki options must be returned as an object.")

        def text_values(key):
            items = value.get(key, ())
            if (
                    not isinstance(items, (tuple, list))
                    or not all(
                        isinstance(item, str) and item.strip()
                        for item in items)):
                raise ValueError(
                    f"{key.replace('_', ' ').title()} must be text values.")
            return tuple(sorted(set(items)))

        if action == "note_types":
            note_types = text_values("note_types")
            current = row["note_var"].get().strip()
            if current not in note_types:
                row["note_var"].set("")
                row["field_var"].set("")
            row["note_box"].configure(
                values=note_types,
                state="readonly")
            row["field_box"].configure(
                values=(),
                state="disabled")
            self.source_exclusion_status.set(
                f"Loaded {len(note_types):,} note types from the deck.")
            if current and current in note_types:
                self._request_learned_filter_options(
                    row,
                    "note_type_details",
                    current)
            self._save_learned_filter_rows()
            return
        if action != "note_type_details":
            raise ValueError(f"Unknown Anki option action: {action}")
        fields = text_values("fields")
        current = row["field_var"].get().strip()
        if current not in fields:
            row["field_var"].set("")
        row["field_box"].configure(
            values=fields,
            state="readonly")
        self.source_exclusion_status.set(
            f"Loaded {len(fields):,} fields from the note type.")
        self._save_learned_filter_rows()

    def _selected_anki_exclusions(self, language_key):
        settings = self._language_filter(language_key)
        if not settings.enabled:
            return ()
        if not settings.sources:
            raise ValueError(
                "Add at least one learned-word deck, note type, and field.")
        if not all(source.complete for source in settings.sources):
            raise ValueError(
                "Finish or remove every incomplete learned-word row.")
        return tuple(
            source.to_exclusion_mapping()
            for source in settings.sources)

    def _manual_input_filter_request(
            self,
            text,
            *,
            make_items_for_characters=False):
        candidates = tuple(
            line.strip()
            for line in text.splitlines()
            if line.strip())
        if not candidates:
            raise ValueError("Enter at least one nonblank line.")
        language = self.get_generation_language()
        request = {
            "text": text,
            "candidates": candidates,
            "anki_exclusions": self._selected_anki_exclusions(
                language.key),
        }
        if make_items_for_characters:
            request.update({
                "language_key": language.model_language_key,
                "make_items_for_characters": True,
                "exclude_anki": self._language_filter(
                    language.key).enabled,
            })
        return request

    def _source_request(self):
        option = self._selected_source_option()
        chunk_size = parse_source_chunk_size(
            self.source_chunk_size.get())
        execution_variable = getattr(
            self,
            "source_execution_label",
            None)
        execution_label = (
            execution_variable.get()
            if hasattr(execution_variable, "get")
            else dict(SOURCE_EXECUTION_OPTIONS)["standard"])
        execution_mode = SOURCE_EXECUTION_KEYS_BY_LABEL.get(
            execution_label)
        if execution_mode is None:
            raise ValueError("Select Standard or Economy processing.")
        model_variable = getattr(self, "source_model_label", None)
        model_label = (
            model_variable.get()
            if hasattr(model_variable, "get")
            else dict(SOURCE_MODEL_OPTIONS)["gpt-5.4-mini"])
        model = SOURCE_MODEL_KEYS_BY_LABEL.get(model_label)
        if model is None:
            raise ValueError("Select a supported source-generation model.")
        model_profile = source_model_profile(model)
        if model_profile.local:
            execution_mode = "standard"
        automatic_variable = getattr(
            self,
            "source_automatic_repair",
            None)
        automatic_repair_preference = bool(
            automatic_variable.get()
            if hasattr(automatic_variable, "get")
            else False)
        automatic_repair = (
            automatic_repair_preference
            and execution_mode == "standard")
        if execution_mode == "economy" or model_profile.local:
            concurrency = (
                model_profile.recommended_concurrency
                if model_profile.local
                else 1)
            request_stagger_ms = 0
        else:
            concurrency = parse_source_chunk_size(
                self.source_concurrency.get(),
                maximum=64)
            request_stagger_ms = parse_request_stagger_ms(
                self.source_request_stagger_ms.get())
        context_mode = SOURCE_CONTEXT_KEYS_BY_LABEL.get(
            self.source_context_label.get())
        if context_mode is None:
            raise ValueError("Select a source context option.")
        use_source_examples_variable = getattr(
            self,
            "source_use_source_examples",
            None)
        use_source_examples_preference = bool(
            use_source_examples_variable.get()
            if hasattr(use_source_examples_variable, "get")
            else False)
        selected_source_directions = (
            self._selected_source_direction_keys()
            if hasattr(self, "source_card_direction_variables")
            else None)
        protocol_variable = getattr(
            self,
            "source_request_protocol_label",
            None)
        protocol_label = (
            protocol_variable.get()
            if hasattr(protocol_variable, "get")
            else dict(SOURCE_PROTOCOL_OPTIONS)["v10"])
        request_protocol = SOURCE_PROTOCOL_KEYS_BY_LABEL.get(
            protocol_label)
        if request_protocol is None:
            raise ValueError("Select a source request protocol.")
        reasoning_variable = getattr(
            self,
            "source_reasoning_label",
            None)
        reasoning_label = (
            reasoning_variable.get()
            if hasattr(reasoning_variable, "get")
            else dict(SOURCE_REASONING_OPTIONS)["low"])
        reasoning_effort = SOURCE_REASONING_KEYS_BY_LABEL.get(
            reasoning_label)
        if reasoning_effort is None:
            raise ValueError("Select a source reasoning effort.")
        if request_protocol == "v8":
            reasoning_effort = "low"
        if model_profile.local:
            request_protocol = "v10"
            reasoning_effort = "none"
        source_prefix_token_limit = None
        prefix_enabled_variable = getattr(
            self,
            "source_limit_to_prefix",
            None)
        prefix_enabled = bool(
            prefix_enabled_variable.get()
            if hasattr(prefix_enabled_variable, "get")
            else False)
        if prefix_enabled:
            prefix_limit_variable = getattr(
                self,
                "source_prefix_token_limit",
                None)
            if not hasattr(prefix_limit_variable, "get"):
                raise ValueError(
                    "Enter the number of running source words to include.")
            source_prefix_token_limit = parse_source_prefix_token_limit(
                prefix_limit_variable.get())
        selected_language_key = option.source_language_key
        language_variable = getattr(self, "source_language_label", None)
        if language_variable is not None:
            selected_language_name = language_variable.get().strip()
            if selected_language_name:
                selected_language_key = self._language_key_for_name(
                    selected_language_name)
        if not selected_language_key:
            raise ValueError("Select the source language and historical era.")
        pipelines = self.get_pipeline_configs()
        if selected_source_directions is None:
            selected_source_directions = (
                tuple(
                    card.direction_key
                    for card in pipeline_store.get_enabled_cards(
                        pipelines[0]))
                if pipelines
                else ("context",))
        if not selected_source_directions:
            raise ValueError("Select at least one source card type.")
        use_source_examples = bool(
            use_source_examples_preference
            and context_mode != "none"
            and "context" in selected_source_directions)
        if use_source_examples:
            context_mode = "sentence"
        if selected_language_key:
            source_pipelines = []
            for pipeline in pipelines:
                settings = pipeline_store.get_language_settings(
                    pipeline,
                    selected_language_key)
                source_cards = []
                for card in settings.cards:
                    enabled = (
                        card.direction_key
                        in selected_source_directions)
                    effective_fields = (
                        pipeline_store.get_effective_fields(
                            settings,
                            card))
                    if (
                            enabled
                            and card.direction_key == "context"
                            and not effective_fields):
                        # Dedicated source-sentence notes do not consume a
                        # definition field. Keep the transient pipeline valid
                        # without asking the provider for this placeholder.
                        effective_fields = (
                            pipeline_store.FieldSetting(
                                "dictionary_meaning",
                                "english"),
                        )
                    source_cards.append(replace(
                        card,
                        enabled=enabled,
                        fields=tuple(effective_fields)))
                settings = replace(
                    settings,
                    cards=tuple(source_cards),
                    target_deck=f"Vocabulary from {option.name}",
                    separate_target_decks=False,
                    share_field_settings=False)
                settings_by_key = {
                    item.language_key: item
                    for item in pipeline.language_settings
                }
                settings_by_key[settings.language_key] = settings
                source_pipelines.append(
                    pipeline_store.replace_active_language_settings(
                        pipeline,
                        settings,
                        tuple(settings_by_key.values()),
                        active_language_key=(
                            selected_language_key)))
            pipelines = pipeline_store.validate_pipelines(
                source_pipelines)
        exclusions = self._selected_anki_exclusions(
            selected_language_key)
        return {
            "source_key": option.key,
            "source_name": option.name,
            "source_language_key": selected_language_key,
            "chunk_size": chunk_size,
            "source_prefix_token_limit": source_prefix_token_limit,
            "context_mode": context_mode,
            "concurrency": concurrency,
            "request_stagger_ms": request_stagger_ms,
            "max_transient_retries": (
                model_profile.default_transient_retries),
            "model": model,
            "request_protocol": request_protocol,
            "reasoning_effort": reasoning_effort,
            "execution_mode": execution_mode,
            "automatic_repair": automatic_repair,
            "max_automatic_repairs": (
                model_profile.max_automatic_repairs),
            "allow_web_search": (
                False
                if model_profile.local
                else bool(
                    getattr(
                        self,
                        "source_allow_web_search",
                        False).get()
                    if hasattr(
                        getattr(self, "source_allow_web_search", None),
                        "get")
                    else False)),
            "use_source_for_example_sentences": use_source_examples,
            "include_source_context_nuance": bool(
                use_source_examples
                and "context" in selected_source_directions
                and (
                    getattr(
                        self,
                        "source_include_context_nuance",
                        False).get()
                    if hasattr(
                        getattr(
                            self,
                            "source_include_context_nuance",
                            None),
                        "get")
                    else False)),
            "source_card_directions": list(
                selected_source_directions),
            "separate_source_decks": bool(
                getattr(
                    self,
                    "source_separate_decks",
                    False).get()
                if hasattr(
                    getattr(self, "source_separate_decks", None),
                    "get")
                else False),
            "pipeline": pipelines[0] if pipelines else None,
            "pipelines": pipelines,
            "anki_exclusions": exclusions,
            "exclude_anki": bool(exclusions),
            "output_deck_name": f"Vocabulary from {option.name}",
            "keep_imported_deck": True,
            "move_cards_after_import": False,
            "delete_imported_deck": False,
        }

    def _schedule_source_estimate(self, *_args):
        if not hasattr(self, "source_estimate_price"):
            return
        # Invalidate every already-running estimator immediately, rather than
        # waiting for the debounced replacement to start. Otherwise an older
        # result can briefly repopulate the price and enable authorization
        # during this 250 ms window.
        self.source_estimate_generation += 1
        self.source_estimate_result = None
        self.source_paid_authorized.set(False)
        self._update_source_generate_button_state()
        if self.source_estimate_after_id is not None:
            try:
                self.root.after_cancel(
                    self.source_estimate_after_id)
            except tk.TclError:
                pass
        self.source_estimate_price.set("Calculating…")
        self.source_estimate_detail.set(
            "Offline estimate; this does not contact OpenAI.")
        self.source_estimate_after_id = self.root.after(
            250,
            self._refresh_source_estimate)

    def _refresh_source_estimate(self):
        self.source_estimate_after_id = None
        if self.source_estimator is None:
            self.source_estimate_price.set("Estimate unavailable")
            self.source_estimate_detail.set(
                "The source estimator is not connected in this build.")
            self._update_source_generate_button_state()
            return
        try:
            request = self._source_request()
        except (OSError, ValueError) as error:
            self.source_estimate_price.set("Check the options")
            self.source_estimate_detail.set(str(error))
            self._update_source_generate_button_state()
            return

        self.source_estimate_generation += 1
        generation = self.source_estimate_generation
        self.source_estimate_pending.add(generation)

        def estimate_in_background():
            try:
                value = self.source_estimator(request)
            except Exception as error:
                self.source_estimate_result_queue.put(
                    (generation, "error", error))
            else:
                self.source_estimate_result_queue.put(
                    (generation, "success", value))

        threading.Thread(
            target=estimate_in_background,
            daemon=True).start()
        if not self.source_estimate_polling:
            self.source_estimate_polling = True
            self.root.after(
                self.POLL_INTERVAL_MS,
                self._poll_source_estimate)

    def _poll_source_estimate(self):
        try:
            generation, outcome, value = (
                self.source_estimate_result_queue.get_nowait())
        except queue.Empty:
            if self.source_estimate_pending:
                self.root.after(
                    self.POLL_INTERVAL_MS,
                    self._poll_source_estimate)
            else:
                self.source_estimate_polling = False
            return
        self.source_estimate_pending.discard(generation)
        if generation != self.source_estimate_generation:
            if self.source_estimate_pending:
                self.root.after(
                    self.POLL_INTERVAL_MS,
                    self._poll_source_estimate)
            else:
                self.source_estimate_polling = False
            return
        if outcome == "error":
            self.source_estimate_result = None
            self.source_estimate_price.set("Estimate failed")
            self.source_estimate_detail.set(str(value))
        else:
            self.source_estimate_result = value
            price, detail = format_source_estimate(value)
            source_examples_enabled = bool(
                self.source_use_source_examples.get())
            detail += (
                "\nExample-sentence mode: "
                + (
                    "ONE CARD PER EXACT SOURCE SENTENCE; original text is "
                    "inserted locally, while selected word cards request "
                    "lexical fields without generated examples."
                    if source_examples_enabled
                    else (
                        "MODEL-GENERATED EXAMPLES for every sense; source "
                        "passages are used only to identify meaning.")
                ))
            self.source_estimate_price.set(price)
            self.source_estimate_detail.set(detail)
            if (
                    self._estimate_candidate_count(value) == 0
                    and self._estimate_request_count(value) == 0
                    and self._estimate_source_sentence_count(value) == 0
                    and self.source_exclude_enabled.get()
                    and not self.source_zero_notice_shown):
                self.source_zero_notice_shown = True
                messagebox.showinfo(
                    "No new vocabulary",
                    (
                        "Every candidate word is already present in the "
                        "selected Anki field. No OpenAI request or deck will "
                        "be created."
                    ),
                    parent=self.root)
        self._update_source_generate_button_state()
        if self.source_estimate_pending:
            self.root.after(
                self.POLL_INTERVAL_MS,
                self._poll_source_estimate)
        else:
            self.source_estimate_polling = False

    @staticmethod
    def _estimate_candidate_count(estimate):
        if estimate is None:
            return None
        if isinstance(estimate, dict):
            return estimate.get("candidate_count")
        return getattr(estimate, "candidate_count", None)

    @staticmethod
    def _estimate_request_count(estimate):
        if estimate is None:
            return None
        if isinstance(estimate, dict):
            return estimate.get("request_count")
        return getattr(estimate, "request_count", None)

    @staticmethod
    def _estimate_source_sentence_count(estimate):
        if estimate is None:
            return 0
        if isinstance(estimate, dict):
            return estimate.get("source_sentence_card_count", 0)
        return getattr(
            estimate,
            "source_sentence_card_count",
            0)

    @classmethod
    def _estimate_has_work(cls, estimate):
        candidate_count = cls._estimate_candidate_count(estimate)
        request_count = cls._estimate_request_count(estimate)
        return (
            candidate_count != 0
            or request_count not in (None, 0)
            or cls._estimate_source_sentence_count(estimate) > 0)

    @classmethod
    def _estimate_requires_paid_authorization(cls, estimate):
        request_count = cls._estimate_request_count(estimate)
        if request_count is not None:
            return request_count > 0
        return cls._estimate_candidate_count(estimate) != 0

    @staticmethod
    def _estimate_limit_warning(estimate):
        if estimate is None:
            return None
        if isinstance(estimate, dict):
            return estimate.get("request_limit_warning")
        return getattr(estimate, "request_limit_warning", None)

    def _update_source_generate_button_state(self):
        if not hasattr(self, "source_generate_button"):
            return
        available = (
            self.source_generate_callback is not None
            and self.source_estimate_result is not None
            and self._estimate_has_work(self.source_estimate_result)
            and not self._estimate_limit_warning(
                self.source_estimate_result)
            and (
                not self._estimate_requires_paid_authorization(
                    self.source_estimate_result)
                or self.source_paid_authorized.get())
            and not self.source_action_in_progress)
        self.source_generate_button.configure(
            state=tk.NORMAL if available else tk.DISABLED)

    def _update_source_codex_button_state(self):
        available = (
            self.source_codex_callback is not None
            and self.source_codex_authorized.get()
            and not self.source_action_in_progress)
        self.source_codex_button.configure(
            state=tk.NORMAL if available else tk.DISABLED)

    def _start_source_generation(self):
        requires_paid_authorization = (
            self._estimate_requires_paid_authorization(
                self.source_estimate_result))
        if (
                requires_paid_authorization
                and not self.source_paid_authorized.get()):
            messagebox.showwarning(
                "Paid requests not authorized",
                "Tick the paid-request authorization before generating.",
                parent=self.root)
            return
        try:
            request = self._source_request()
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Cannot start source generation",
                str(error),
                parent=self.root)
            return
        if not self._estimate_has_work(self.source_estimate_result):
            messagebox.showinfo(
                "No new vocabulary",
                (
                    "Every candidate word is already present in the selected "
                    "Anki field. No OpenAI request or deck was created."
                ),
                parent=self.root)
            return
        request["paid_confirmed"] = bool(
            requires_paid_authorization
            and self.source_paid_authorized.get())
        request["estimate"] = self.source_estimate_result
        self._dispatch_source_action(
            "generate",
            self.source_generate_callback,
            request)

    def _choose_source_file(self):
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a source file",
            filetypes=(
                ("Supported sources", "*.pdf *.txt *.md"),
                ("PDF files", "*.pdf"),
                ("Text files", "*.txt *.md"),
                ("All files", "*.*"),
            ))
        if not selected:
            return
        path = Path(selected)
        self.source_file_path.set(str(path))
        if not self.source_file_title.get().strip():
            self.source_file_title.set(path.stem)

    def _language_key_for_name(self, name):
        for language in pipeline_store.list_languages():
            if language.name == name:
                return language.key
        raise ValueError("Select a valid language and historical era.")

    def _start_source_preparation(self):
        path = Path(
            self.source_file_path.get().strip()).expanduser()
        if not path.is_file():
            messagebox.showerror(
                "Source file not found",
                "Choose an existing PDF, TXT, or Markdown file.",
                parent=self.root)
            return
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            messagebox.showerror(
                "Unsupported source file",
                "Choose a PDF, TXT, or Markdown file.",
                parent=self.root)
            return
        title = self.source_file_title.get().strip()
        if not title:
            messagebox.showerror(
                "Source title required",
                "Enter the title that should appear in the source list.",
                parent=self.root)
            return
        try:
            language_key = self._language_key_for_name(
                self.source_file_language.get())
        except ValueError as error:
            messagebox.showerror(
                "Language required",
                str(error),
                parent=self.root)
            return
        self._dispatch_source_action(
            "prepare",
            self.source_prepare_callback,
            {
                "path": str(path.resolve()),
                "title": title,
                "language_key": language_key,
                "prefer_cuda": self.source_file_use_gpu.get(),
                "generate_cards": False,
            })

    def _start_source_codex_retrieval(self):
        request_text = self.source_codex_text.get(
            "1.0",
            tk.END).strip()
        if not request_text:
            messagebox.showwarning(
                "Describe the source",
                "Enter the edition and text you want Codex to retrieve.",
                parent=self.root)
            return
        source_title = self.source_codex_title.get().strip()
        if not source_title:
            messagebox.showwarning(
                "Source title required",
                "Enter the title that should appear in the prepared list.",
                parent=self.root)
            return
        if not self.source_codex_authorized.get():
            messagebox.showwarning(
                "Retrieval not authorized",
                "Tick the retrieval authorization before starting Codex.",
                parent=self.root)
            return
        try:
            language_key = self._language_key_for_name(
                self.source_codex_language.get())
        except ValueError as error:
            messagebox.showerror(
                "Language required",
                str(error),
                parent=self.root)
            return
        self._dispatch_source_action(
            "codex",
            self.source_codex_callback,
            {
                "request": request_text,
                "description": request_text,
                "title": source_title,
                "source_title": source_title,
                "language_key": language_key,
                "retrieval_authorized": True,
                "generate_cards": False,
            })

    @staticmethod
    def _dispatch_status_message(value, fallback):
        if isinstance(value, dict):
            message = value.get("message")
        else:
            message = getattr(value, "message", None)
        return str(message or fallback)

    def _refresh_source_dispatch_status(self):
        if self.source_dispatch_status_loader is None:
            self.source_dispatch_status.set(
                "Emergency control unavailable")
            return
        try:
            value = self.source_dispatch_status_loader({})
        except Exception as error:
            self.source_dispatch_status.set(
                f"Could not read dispatch status · {error}")
            return
        status = (
            value.get("status")
            if isinstance(value, dict)
            else None)
        message = self._dispatch_status_message(value, "Ready")
        self.source_dispatch_status.set(
            f"Paused · {message}"
            if status == "paused" and not message.startswith("Paused")
            else message)

    def _pause_source_dispatch(self):
        if self.source_pause_callback is None:
            self.source_dispatch_status.set(
                "Emergency control unavailable")
            return
        self.source_dispatch_status.set("Pausing…")
        try:
            value = self.source_pause_callback({})
        except Exception as error:
            self.source_dispatch_status.set(
                f"Pause failed · {error}")
            return
        message = self._dispatch_status_message(
            value,
            "No queued paid request will be sent.")
        self.source_dispatch_status.set(f"Paused · {message}")
        self.source_action_status.set(
            "Emergency pause is active. Completed responses remain saved.")
        self._refresh_source_jobs(show_errors=False)

    def _resume_source_dispatch(self):
        if self.source_resume_callback is None:
            self.source_dispatch_status.set(
                "Emergency control unavailable")
            return
        self.source_dispatch_status.set("Connecting…")
        self._dispatch_source_action(
            "resume",
            self.source_resume_callback,
            {})

    def _dispatch_source_action(self, action, callback, request):
        if callback is None:
            messagebox.showerror(
                "Feature not connected",
                "This source action is not connected in this build.",
                parent=self.root)
            return
        if self.source_action_in_progress:
            messagebox.showwarning(
                "Source action already running",
                "Wait for the current source action to finish.",
                parent=self.root)
            return
        self._set_source_action_busy(True)
        labels = {
            "generate": "Starting paid source-generation jobs…",
            "prepare": "Preparing and tokenizing the source…",
            "codex": "Codex is retrieving and preparing the source…",
            "retry": "Retrying the explicitly selected failed jobs…",
            "resume": "Connecting…",
            "inspect": "Loading saved request and response details…",
            "validation_accept": (
                "Saving the human review and rechecking the response…"),
        }
        self.source_action_status.set(
            labels.get(action, "Starting source action…"))

        def action_in_background():
            try:
                result = callback(request)
            except Exception as error:
                self.source_action_result_queue.put(
                    (action, "error", error))
            else:
                self.source_action_result_queue.put(
                    (action, "success", result))

        threading.Thread(
            target=action_in_background,
            daemon=True).start()
        self.root.after(
            self.POLL_INTERVAL_MS,
            self._poll_source_action_result)

    def _set_source_action_busy(self, busy):
        self.source_action_in_progress = busy
        if hasattr(self, "source_prepare_button"):
            self.source_prepare_button.configure(
                state=(
                    tk.DISABLED
                    if busy or self.source_prepare_callback is None
                    else tk.NORMAL))
        if hasattr(self, "source_codex_button"):
            self._update_source_codex_button_state()
        if hasattr(self, "source_retry_selected_button"):
            retry_state = (
                tk.DISABLED
                if busy or self.source_retry_callback is None
                else tk.NORMAL)
            self.source_retry_selected_button.configure(
                state=retry_state)
            self.source_retry_all_button.configure(
                state=retry_state)
        if hasattr(self, "source_resume_button"):
            self.source_resume_button.configure(
                state=(
                    tk.DISABLED
                    if busy or self.source_resume_callback is None
                    else tk.NORMAL))
        self._update_source_problem_button_state()
        self._update_validation_problem_button_states()
        self._update_source_generate_button_state()

    def _poll_source_action_result(self):
        try:
            action, outcome, value = (
                self.source_action_result_queue.get_nowait())
        except queue.Empty:
            self._refresh_source_jobs(show_errors=False)
            self.root.after(
                1000,
                self._poll_source_action_result)
            return
        self._set_source_action_busy(False)
        if outcome == "error":
            if action == "resume":
                error_text = str(value)
                error_name = type(value).__name__.lower()
                combined = f"{error_name} {error_text}".lower()
                if "timeout" in combined:
                    status = "Connection timeout"
                elif any(
                        marker in combined
                        for marker in (
                            "connection",
                            "network",
                            "socket",
                            "dns",
                            "unreachable",
                        )):
                    status = "No connection"
                else:
                    status = f"Resume failed · {error_text}"
                self.source_dispatch_status.set(status)
                self.source_action_status.set(status)
                self._refresh_source_jobs(show_errors=False)
                return
            self.source_action_status.set(
                f"Source action failed · {value}")
            messagebox.showerror(
                "Source action failed",
                str(value),
                parent=self.root)
            self._refresh_source_jobs(show_errors=False)
            if action == "generate":
                # The backend fails closed when the source plan, prompt,
                # schema, model, learned-word snapshot, or displayed estimate
                # changed after authorization. Always refresh and require a
                # new tick after a failed generation attempt.
                self.source_paid_authorized.set(False)
                self._schedule_source_estimate()
            return

        if action == "resume":
            self._refresh_source_jobs(show_errors=False)
            message = self._dispatch_status_message(
                value,
                "Successfully resumed")
            self.source_dispatch_status.set(message)
            self.source_action_status.set(message)
            return

        if action == "inspect":
            self._show_source_job_inspection(value)
            self.source_action_status.set(
                "Loaded the saved request and response.")
            return
        if action == "validation_accept":
            self._refresh_source_jobs(show_errors=False)
            message = (
                value.get("message")
                if isinstance(value, dict)
                else None)
            context = getattr(
                self,
                "source_validation_problem_context",
                None)
            if isinstance(context, dict):
                record = context["record"]
                try:
                    inspection = self.source_inspect_callback({
                        "job_id": record["job_id"]})
                except Exception:
                    inspection = dict(context["inspection"])
                    if isinstance(value, dict) and isinstance(
                            value.get("validation"), dict):
                        inspection["validation"] = value["validation"]
                self._show_source_job_inspection(
                    inspection,
                    record=record)
                self._refresh_validation_problem_dialog(
                    inspection,
                    record=record)
            self.source_action_status.set(
                message or "Saved the manual validation decision.")
            messagebox.showinfo(
                "Validation review saved",
                message or (
                    "The selected validation decision was saved and the "
                    "retained response was checked again."),
                parent=(
                    self.source_validation_problem_dialog
                    if getattr(
                        self,
                        "source_validation_problem_dialog",
                        None) is not None
                    else self.root))
            self._schedule_source_estimate()
            return
        if action in {"prepare", "codex"}:
            self._load_source_catalogue()
        self._refresh_source_jobs(show_errors=False)
        if isinstance(value, dict):
            message = value.get("message")
        else:
            message = getattr(value, "message", None)
        self.source_action_status.set(
            message or {
                "generate": (
                    "Generation coordinator finished. Inspect Jobs & "
                    "Failures for per-chunk results."
                ),
                "prepare": "Source preparation finished.",
                "codex": "Codex retrieval and source preparation finished.",
                "retry": "Selected retries finished.",
                "resume": "Successfully resumed.",
            }.get(action, "Source action finished."))
        requires_attention = (
            isinstance(value, dict)
            and bool(value.get("requires_attention")))
        no_new_vocabulary = (
            isinstance(value, dict)
            and bool(value.get("no_new_vocabulary")))
        if no_new_vocabulary and action == "generate":
            messagebox.showinfo(
                "No new vocabulary",
                (
                    f"{message or 'No new vocabulary remains.'}\n\n"
                    "No OpenAI request was sent and no Anki deck was "
                    "created."
                ),
                parent=self.root)
        if requires_attention and action in {"generate", "retry"}:
            self.source_notebook.select(self.source_jobs_page)
            messagebox.showwarning(
                "Source requests need attention",
                (
                    f"{message or 'One or more source requests failed.'}\n\n"
                    "Nothing invalid was retried automatically. In Jobs & "
                    "Failures you can inspect the retained response, retry "
                    "selected requests, or retry all failed requests."
                ),
                parent=self.root)
        self._schedule_source_estimate()

    def _refresh_source_jobs(self, *, show_errors=True):
        if not getattr(self, "_source_jobs_page_built", False):
            return
        if self.source_jobs_loader is None:
            if show_errors:
                self.source_action_status.set(
                    "Saved source jobs are not connected in this build.")
            return
        try:
            jobs = self.source_jobs_loader()
            if isinstance(jobs, dict):
                jobs = jobs.get("jobs", ())
            jobs = tuple(jobs)
        except Exception as error:
            if show_errors:
                messagebox.showerror(
                    "Could not load source jobs",
                    str(error),
                    parent=self.root)
            self.source_action_status.set(
                f"Could not load source jobs · {error}")
            return
        previously_selected = {
            record["job_id"]
            for record in self._selected_source_jobs()}
        for item_id in self.source_jobs_tree.get_children():
            self.source_jobs_tree.delete(item_id)
        self.source_job_by_tree_id = {}
        display_rows = []
        groups = group_source_job_rows(jobs)
        for group_index, (parent_job_id, group_rows) in enumerate(groups):
            normalized_rows = []
            for child_index, job in enumerate(group_rows):
                job_id = str(_source_job_value(
                    job,
                    "job_id",
                    _source_job_value(
                        job,
                        "id",
                        f"{group_index}-{child_index}")))
                source = str(_source_job_value(
                    job,
                    "source_name",
                    _source_job_value(job, "source", "")))
                chunk = str(_source_job_value(
                    job,
                    "chunk_label",
                    _source_job_value(
                        job,
                        "chunk",
                        _source_job_value(job, "chunk_index", ""))))
                worker = str(_source_job_value(
                    job,
                    "worker",
                    _source_job_value(job, "worker_id", "—")))
                status = str(_source_job_value(job, "status", "unknown"))
                attempts = str(_source_job_value(
                    job,
                    "attempts",
                    _source_job_value(job, "attempt_count", 0)))
                detail_value = _source_job_value(
                    job,
                    "detail",
                    _source_job_value(
                        job,
                        "error",
                        _source_job_value(job, "message", "")))
                if isinstance(detail_value, (dict, list, tuple)):
                    detail = _source_inspection_json(detail_value)
                else:
                    detail = str(detail_value)
                detail_kind = str(_source_job_value(
                    job,
                    "detail_kind",
                    ""))
                has_validation_error = bool(_source_job_value(
                    job,
                    "has_validation_error",
                    detail_kind == "validation_error"))
                normalized_rows.append({
                    "job_id": job_id,
                    "parent_job_id": parent_job_id,
                    "source": source,
                    "chunk": chunk,
                    "worker": worker,
                    "status": status,
                    "attempts": attempts,
                    "detail": detail,
                    "detail_kind": detail_kind,
                    "has_validation_error": has_validation_error,
                    "record": job,
                })

            group_label = source_job_group_label(
                parent_job_id,
                group_index + 1)
            source = next(
                (
                    row["source"]
                    for row in normalized_rows
                    if row["source"]),
                "—")
            request_count = sum(
                row["chunk"] != "Deck"
                for row in normalized_rows)
            statuses = {
                row["status"].lower()
                for row in normalized_rows}
            if statuses & {
                    "failed",
                    "invalid",
                    "invalid_response",
                    "connection_failed",
                    "cancelled"}:
                group_status = "needs attention"
            elif "running" in statuses:
                group_status = "running"
            elif statuses and statuses <= {"completed", "succeeded"}:
                group_status = "completed"
            elif "pending" in statuses:
                group_status = "pending"
            else:
                group_status = "mixed"
            group_status_tag = {
                "needs attention": "failed",
                "running": "running",
                "completed": "completed",
            }.get(group_status)
            group_detail = (
                f"{request_count:,} OpenAI request chunk"
                f"{'' if request_count == 1 else 's'}; "
                "the Deck packaging/import stage is listed last.")
            group_tree_id = f"source_job_group_{group_index}"
            group_display = {
                "job": group_label,
                "source": source,
                "chunk": f"{len(normalized_rows):,} stages",
                "worker": "—",
                "status": group_status,
                "attempts": "—",
                "detail": group_detail,
            }
            display_rows.append(group_display)
            self.source_jobs_tree.insert(
                "",
                tk.END,
                iid=group_tree_id,
                text=group_label,
                open=True,
                values=(
                    source,
                    group_display["chunk"],
                    "—",
                    group_status,
                    "—",
                    group_detail,
                ),
                tags=tuple(
                    tag
                    for tag in ("job_group", group_status_tag)
                    if tag))
            stripe = f"job_{'even' if group_index % 2 == 0 else 'odd'}"
            for child_index, row in enumerate(normalized_rows):
                tree_id = f"source_job_{group_index}_{child_index}"
                display_rows.append({
                    "job": "",
                    **{
                        column: row[column]
                        for column in SOURCE_JOB_COLUMNS
                    },
                })
                self.source_jobs_tree.insert(
                    group_tree_id,
                    tk.END,
                    iid=tree_id,
                    text="",
                    values=tuple(
                        row[column]
                        for column in SOURCE_JOB_COLUMNS),
                    tags=tuple(
                        dict.fromkeys((
                            row["status"].lower(),
                            stripe,
                            *(
                                ("validation_error",)
                                if row["has_validation_error"]
                                else ()),
                        ))))
                self.source_job_by_tree_id[tree_id] = row
                if row["job_id"] in previously_selected:
                    self.source_jobs_tree.selection_add(tree_id)
        self.source_job_display_rows = tuple(display_rows)
        self._resize_source_job_columns()
        self._source_job_selection_changed()
        self.source_action_status.set(
            (
                f"{len(jobs):,} saved stages loaded across "
                f"{len(groups):,} deck-generation "
                f"{'job' if len(groups) == 1 else 'jobs'}."
            ))

    def _selected_source_jobs(self):
        return tuple(
            self.source_job_by_tree_id[item_id]
            for item_id in self.source_jobs_tree.selection()
            if item_id in self.source_job_by_tree_id)

    def _source_job_selection_changed(self, _event=None):
        selected_ids = tuple(self.source_jobs_tree.selection())
        if len(selected_ids) != 1:
            self.source_latest_detail_heading.set("FULL LATEST DETAIL")
            self._set_source_latest_detail(
                (
                    "Select one request chunk or Deck stage above to read its "
                    "complete latest detail here."
                    if not selected_ids
                    else (
                        f"{len(selected_ids):,} rows are selected. Select one "
                        "row to show its complete latest detail."
                    )))
            self._update_source_problem_button_state()
            return
        item_id = selected_ids[0]
        record = self.source_job_by_tree_id.get(item_id)
        if record is None:
            values = self.source_jobs_tree.item(item_id, "values")
            detail = values[-1] if values else ""
            self.source_latest_detail_heading.set("JOB SUMMARY")
        else:
            detail = record["detail"]
            stage = record["chunk"] or record["job_id"]
            self.source_latest_detail_heading.set(
                (
                    f"VALIDATION ERROR · {stage}"
                    if record.get("has_validation_error")
                    else f"FULL LATEST DETAIL · {stage}"))
        self._set_source_latest_detail(
            detail or "No detail has been saved for this stage.",
            validation_error=(
                record is not None
                and record.get("has_validation_error")))
        self._update_source_problem_button_state()

    def _update_source_problem_button_state(self):
        if not hasattr(self, "source_view_problems_button"):
            return
        selected = self._selected_source_jobs()
        available = (
            len(selected) == 1
            and selected[0]["chunk"] != "Deck"
            and not selected[0]["job_id"].endswith("::finalize")
            and self.source_inspect_callback is not None
            and not self.source_action_in_progress)
        self.source_view_problems_button.configure(
            state=tk.NORMAL if available else tk.DISABLED)

    def _inspect_selected_source_job(self):
        selected = self._selected_source_jobs()
        if len(selected) != 1:
            messagebox.showinfo(
                "Select one stage",
                (
                    "Select exactly one request chunk or Deck stage to "
                    "inspect. A Job heading represents the whole run and "
                    "cannot itself be inspected."
                ),
                parent=self.root)
            return
        record = selected[0]
        if self.source_inspect_callback is None:
            self._show_source_job_inspection(
                record["record"],
                record=record)
            return
        try:
            value = self.source_inspect_callback({
                "job_id": record["job_id"]})
        except Exception as error:
            messagebox.showerror(
                "Could not inspect source job",
                str(error),
                parent=self.root)
            return
        self._show_source_job_inspection(
            value,
            record=record)
        self.source_action_status.set(
            "Loaded the saved request and response.")

    @staticmethod
    def _replace_readonly_text(widget, text):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.yview_moveto(0.0)
        widget.configure(state=tk.DISABLED)

    def _show_source_job_inspection(self, value, *, record=None):
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        self.source_last_inspection = value
        self.source_last_inspection_record = record
        request_text, response_text = source_job_inspection_texts(value)
        self._replace_readonly_text(
            self.source_job_request_inspection,
            request_text)
        self._replace_readonly_text(
            self.source_job_response_inspection,
            response_text)
        backend_scope = (
            value.get("scope")
            if isinstance(value, dict)
            else None)
        if (
                isinstance(backend_scope, dict)
                and isinstance(backend_scope.get("summary"), str)):
            self.source_inspection_scope.set(
                backend_scope["summary"])
        elif record is None:
            self.source_inspection_scope.set(
                (
                    "Showing one saved stage. The Request tab contains its "
                    "input; the Response tab contains saved attempts or its "
                    "packaging/import result."
                ))
        else:
            self.source_inspection_scope.set(
                source_job_inspection_scope(record))
        self.source_job_inspection_notebook.select(
            self.source_job_response_inspection.master)

    def _open_selected_source_problems(self):
        selected = self._selected_source_jobs()
        if len(selected) != 1:
            messagebox.showinfo(
                "Select one request chunk",
                (
                    "Select exactly one OpenAI request chunk. Problems belong "
                    "to individual responses, not to a whole Job heading or "
                    "the Deck packaging/import stage."
                ),
                parent=self.root)
            return
        record = selected[0]
        if (
                record["chunk"] == "Deck"
                or record["job_id"].endswith("::finalize")):
            messagebox.showinfo(
                "Select an OpenAI request chunk",
                (
                    "The Deck row summarizes packaging/import. Select an "
                    "invalid numbered request chunk to review card problems."
                ),
                parent=self.root)
            return
        if self.source_inspect_callback is None:
            messagebox.showerror(
                "Validation review is not connected",
                "This build cannot load retained response problems.",
                parent=self.root)
            return
        try:
            inspection = self.source_inspect_callback({
                "job_id": record["job_id"]})
        except Exception as error:
            messagebox.showerror(
                "Could not load validation problems",
                str(error),
                parent=self.root)
            return
        if hasattr(inspection, "to_dict"):
            inspection = inspection.to_dict()
        self._show_source_job_inspection(
            inspection,
            record=record)
        report = inspection_validation_report(inspection)
        if not isinstance(report, dict):
            messagebox.showinfo(
                "No validation report",
                (
                    "This saved stage has no structured validation report. "
                    "Its raw attempt remains available in the Response tab."
                ),
                parent=self.root)
            return
        self._show_validation_problem_dialog(
            record,
            inspection)

    def _show_validation_problem_dialog(self, record, inspection):
        existing = getattr(
            self,
            "source_validation_problem_dialog",
            None)
        if existing is not None:
            try:
                existing.destroy()
            except tk.TclError:
                pass

        dialog = tk.Toplevel(self.root)
        dialog.title(
            f"Validation problems · {record['source']} · {record['chunk']}")
        dialog.configure(background=self.WINDOW_BACKGROUND)
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        (
            dialog_width,
            dialog_height,
            dialog_x,
            dialog_y,
        ) = source_validation_dialog_geometry(
            screen_width,
            screen_height)
        dialog.geometry(
            f"{dialog_width}x{dialog_height}+{dialog_x}+{dialog_y}")
        dialog.minsize(
            min(SOURCE_VALIDATION_DIALOG_MIN_WIDTH, dialog_width),
            min(SOURCE_VALIDATION_DIALOG_MIN_HEIGHT, dialog_height))
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(2, weight=1)
        dialog.protocol(
            "WM_DELETE_WINDOW",
            self._close_validation_problem_dialog)
        self.source_validation_problem_dialog = dialog
        self.source_validation_problem_context = {
            "record": record,
            "inspection": inspection,
            "problems_by_item": {},
        }
        self.source_validation_problem_panes_sized = False

        heading = ttk.Frame(
            dialog,
            padding=(18, 15, 18, 5),
            style="App.TFrame")
        heading.grid(row=0, column=0, sticky="ew")
        heading.columnconfigure(0, weight=1)
        ttk.Label(
            heading,
            text="REVIEW RESPONSE PROBLEMS",
            style="HelpSection.TLabel").grid(
                row=0,
                column=0,
                sticky="w")
        self.source_validation_problem_scope_label = ttk.Label(
            heading,
            text=(
                f"{record['source']} · request chunk {record['chunk']} · "
                "click a problem to see the exact affected card or section"),
            style="Status.TLabel",
            wraplength=max(1, dialog_width - 52),
            justify=tk.LEFT)
        self.source_validation_problem_scope_label.grid(
                row=1,
                column=0,
                sticky="ew",
                pady=(3, 0))

        self.source_validation_problem_summary = tk.StringVar()
        self.source_validation_problem_summary_label = ttk.Label(
            dialog,
            textvariable=self.source_validation_problem_summary,
            style="Status.TLabel",
            wraplength=max(1, dialog_width - 52),
            justify=tk.LEFT)
        self.source_validation_problem_summary_label.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=18,
            pady=(3, 9))

        panes = ttk.Panedwindow(
            dialog,
            orient=tk.VERTICAL)
        panes.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=18)
        self.source_validation_problem_panes = panes

        problem_panel = ttk.Frame(
            panes,
            padding=(0, 0, 0, 7),
            style="App.TFrame")
        problem_panel.columnconfigure(0, weight=1)
        problem_panel.rowconfigure(1, weight=1)
        ttk.Label(
            problem_panel,
            text="PROBLEMS",
            style="HelpSection.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                pady=(0, 5))
        columns = ("decision", "location", "problem")
        self.source_validation_problem_tree = ttk.Treeview(
            problem_panel,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=6,
            style="ValidationProblems.Treeview")
        for column, heading_text, width in (
                ("decision", "Review state", 190),
                ("location", "Affected card / field", 500),
                ("problem", "Why it is invalid", 800)):
            self.source_validation_problem_tree.heading(
                column,
                text=heading_text)
            self.source_validation_problem_tree.column(
                column,
                width=width,
                minwidth=40,
                stretch=False)
        problem_scroll = RoundedScrollbar(
            problem_panel,
            command=self.source_validation_problem_tree.yview,
            background=self.WINDOW_BACKGROUND,
            active=self.ACCENT)
        self.source_validation_problem_tree.configure(
            yscrollcommand=problem_scroll.set)
        self.source_validation_problem_tree.grid(
            row=1,
            column=0,
            sticky="nsew")
        problem_scroll.grid(
            row=1,
            column=1,
            sticky="ns",
            padx=(6, 0))
        self.source_validation_problem_tree.tag_configure(
            "locked",
            foreground=self.ERROR)
        self.source_validation_problem_tree.tag_configure(
            "reviewable",
            foreground=self.WARNING)
        self.source_validation_problem_tree.tag_configure(
            "accepted",
            foreground=self.ACCENT)
        self.source_validation_problem_tree.bind(
            "<<TreeviewSelect>>",
            self._validation_problem_selection_changed,
            add="+")
        self.source_validation_problem_tree.bind(
            "<Configure>",
            self._resize_validation_problem_columns,
            add="+")
        panes.add(problem_panel, weight=3)

        detail_panel = ttk.Frame(
            panes,
            padding=(0, 7, 0, 7),
            style="App.TFrame")
        detail_panel.columnconfigure(0, weight=1)
        detail_panel.rowconfigure(1, weight=1)
        ttk.Label(
            detail_panel,
            text="WHY THE SELECTED ITEM IS INVALID",
            style="HelpSection.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                pady=(0, 5))
        detail_body = ttk.Frame(
            detail_panel,
            style="Panel.TFrame")
        detail_body.grid(
            row=1,
            column=0,
            sticky="nsew")
        self.source_validation_problem_details = (
            self._build_validation_problem_text(detail_body))
        panes.add(detail_panel, weight=4)

        affected_panel = ttk.Frame(
            panes,
            padding=(0, 7, 0, 0),
            style="App.TFrame")
        affected_panel.columnconfigure(0, weight=1)
        affected_panel.rowconfigure(1, weight=1)
        ttk.Label(
            affected_panel,
            text="AFFECTED CARD / RESPONSE SECTION",
            style="HelpSection.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                pady=(0, 5))
        affected_body = ttk.Frame(
            affected_panel,
            style="Panel.TFrame")
        affected_body.grid(
            row=1,
            column=0,
            sticky="nsew")
        self.source_validation_affected_section = (
            self._build_validation_problem_text(affected_body))
        panes.add(affected_panel, weight=4)

        controls = ttk.Frame(
            dialog,
            padding=(18, 10, 18, 16),
            style="App.TFrame")
        controls.grid(
            row=3,
            column=0,
            sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(
            controls,
            text="OPTIONAL AUDIT NOTE",
            style="Status.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                padx=(0, 8))
        self.source_validation_reason = tk.StringVar()
        reason_entry = ttk.Entry(
            controls,
            textvariable=self.source_validation_reason,
            style="App.TEntry")
        reason_entry.grid(
            row=0,
            column=1,
            columnspan=3,
            sticky="ew")
        reason_entry.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")
        button_bar = ttk.Frame(
            controls,
            style="App.TFrame")
        button_bar.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0))
        button_bar.columnconfigure(0, weight=1)
        self.source_validation_accept_selected_button = ttk.Button(
            button_bar,
            text="Accept selected…",
            command=self._accept_selected_validation_problems,
            state=tk.DISABLED,
            style="CompactSecondary.TButton",
            cursor="hand2")
        self.source_validation_accept_selected_button.grid(
            row=0,
            column=1,
            sticky="e")
        self.source_validation_accept_all_button = ttk.Button(
            button_bar,
            text="Accept all…",
            command=self._accept_all_validation_problems,
            state=tk.DISABLED,
            style="CompactSecondary.TButton",
            cursor="hand2")
        self.source_validation_accept_all_button.grid(
            row=0,
            column=2,
            sticky="e",
            padx=(8, 0))
        ttk.Button(
            button_bar,
            text="Close",
            command=self._close_validation_problem_dialog,
            style="CompactSecondary.TButton",
            cursor="hand2").grid(
                row=0,
                column=3,
                sticky="e",
                padx=(8, 0))

        self._refresh_validation_problem_dialog(
            inspection,
            record=record)
        dialog.after_idle(
            self._size_validation_problem_panes)
        dialog.bind(
            "<Configure>",
            self._validation_problem_dialog_resized,
            add="+")
        dialog.grab_set()
        dialog.focus_set()

    def _size_validation_problem_panes(self):
        if getattr(
                self,
                "source_validation_problem_panes_sized",
                False):
            return
        panes = getattr(
            self,
            "source_validation_problem_panes",
            None)
        if panes is None:
            return
        try:
            height = panes.winfo_height()
            if height <= 100:
                dialog = getattr(
                    self,
                    "source_validation_problem_dialog",
                    None)
                if dialog is not None:
                    dialog.after(
                        40,
                        self._size_validation_problem_panes)
                return
            panes.sashpos(0, int(height * 0.36))
            panes.sashpos(1, int(height * 0.68))
            self.source_validation_problem_panes_sized = True
        except tk.TclError:
            pass

    def _validation_problem_dialog_resized(self, event=None):
        dialog = getattr(
            self,
            "source_validation_problem_dialog",
            None)
        if (
                dialog is None
                or (
                    event is not None
                    and event.widget is not dialog)):
            return
        try:
            wraplength = max(1, dialog.winfo_width() - 52)
            self.source_validation_problem_scope_label.configure(
                wraplength=wraplength)
            self.source_validation_problem_summary_label.configure(
                wraplength=wraplength)
            self._resize_validation_problem_columns()
        except (AttributeError, tk.TclError):
            pass

    def _resize_validation_problem_columns(self, event=None):
        tree = getattr(
            self,
            "source_validation_problem_tree",
            None)
        if tree is None:
            return
        available_width = (
            event.width
            if event is not None
            else tree.winfo_width())
        try:
            style = ttk.Style(self.root)
            body_font = style.lookup(
                "ValidationProblems.Treeview",
                "font") or ("DejaVu Sans", 9)
            heading_font = style.lookup(
                "ValidationProblems.Treeview.Heading",
                "font") or ("DejaVu Sans", 9, "bold")

            def measure(value):
                text = str(value)
                return max(
                    int(self.root.tk.call(
                        "font",
                        "measure",
                        body_font,
                        text)),
                    int(self.root.tk.call(
                        "font",
                        "measure",
                        heading_font,
                        text)))

            widths = validation_problem_column_widths(
                measure,
                max(1, available_width))
            for column, width in widths.items():
                tree.column(
                    column,
                    width=width,
                    minwidth=40,
                    stretch=False)
        except tk.TclError:
            pass

    def _build_validation_problem_text(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        text = tk.Text(
            parent,
            height=6,
            wrap=tk.WORD,
            font=("DejaVu Sans Mono", 9),
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            selectbackground="#D7E9E5",
            selectforeground=self.TEXT_PRIMARY,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=10,
            pady=8,
            state=tk.DISABLED)
        scrollbar = RoundedScrollbar(
            parent,
            command=text.yview,
            background=self.PANEL_BACKGROUND,
            active=self.ACCENT)
        text.configure(yscrollcommand=scrollbar.set)
        text.grid(
            row=0,
            column=0,
            sticky="nsew")
        scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
            padx=(5, 0))
        return text

    def _close_validation_problem_dialog(self):
        dialog = getattr(
            self,
            "source_validation_problem_dialog",
            None)
        self.source_validation_problem_dialog = None
        self.source_validation_problem_context = None
        if dialog is not None:
            try:
                dialog.grab_release()
                dialog.destroy()
            except tk.TclError:
                pass

    def _refresh_validation_problem_dialog(
            self,
            inspection,
            *,
            record=None):
        context = getattr(
            self,
            "source_validation_problem_context",
            None)
        if not isinstance(context, dict):
            return
        if record is not None:
            context["record"] = record
        context["inspection"] = inspection
        report = inspection_validation_report(inspection) or {}
        context["report"] = report
        self.source_validation_problem_summary.set(
            validation_report_summary(report))
        self.source_validation_problem_summary_label.configure(
            style=(
                "ValidationDialogSafe.TLabel"
                if not report.get("remaining_problem_count", 0)
                else "ValidationDialogError.TLabel"))

        tree = self.source_validation_problem_tree
        for item_id in tree.get_children():
            tree.delete(item_id)
        context["problems_by_item"] = {}
        for index, problem in enumerate(report.get("problems", ())):
            if problem.get("accepted"):
                decision = "Accepted"
                tag = "accepted"
            elif problem.get("overrideable"):
                decision = "Can accept"
                tag = "reviewable"
            else:
                decision = "Locked"
                tag = "locked"
            item_id = f"problem_{index}"
            tree.insert(
                "",
                tk.END,
                iid=item_id,
                values=(
                    decision,
                    problem.get("location") or "Response",
                    problem.get("title") or problem.get("message") or "Invalid",
                ),
                tags=(tag,))
            context["problems_by_item"][item_id] = problem
        self._resize_validation_problem_columns()

        first = next(iter(context["problems_by_item"]), None)
        if first is None:
            self._replace_readonly_text(
                self.source_validation_problem_details,
                (
                    "This retained response has no current validation "
                    "problems."
                ))
            self._replace_readonly_text(
                self.source_validation_affected_section,
                latest_source_response_text(inspection)
                or "No retained response text is available.")
        else:
            tree.selection_set(first)
            tree.focus(first)
            tree.see(first)
            self._validation_problem_selection_changed()
        self._update_validation_problem_button_states()

    def _validation_problem_selection_changed(self, _event=None):
        context = getattr(
            self,
            "source_validation_problem_context",
            None)
        if not isinstance(context, dict):
            return
        selected = tuple(
            context["problems_by_item"][item_id]
            for item_id in self.source_validation_problem_tree.selection()
            if item_id in context["problems_by_item"])
        if not selected:
            details = "Select one problem to see why it failed validation."
            affected = (
                "Select one problem to see its affected card or response "
                "section.")
        else:
            problem = selected[0]
            details = format_validation_problem_details(problem)
            affected = validation_problem_affected_section(
                context["inspection"],
                problem)
        self._replace_readonly_text(
            self.source_validation_problem_details,
            details)
        self._replace_readonly_text(
            self.source_validation_affected_section,
            affected)
        self._update_validation_problem_button_states()

    def _reviewable_validation_problems(self, *, selected_only):
        context = getattr(
            self,
            "source_validation_problem_context",
            None)
        if not isinstance(context, dict):
            return ()
        if selected_only:
            problem_items = (
                context["problems_by_item"].get(item_id)
                for item_id in self.source_validation_problem_tree.selection())
        else:
            problem_items = context["problems_by_item"].values()
        return tuple(
            problem
            for problem in problem_items
            if (
                isinstance(problem, dict)
                and problem.get("overrideable")
                and not problem.get("accepted")))

    def _update_validation_problem_button_states(self):
        dialog = getattr(
            self,
            "source_validation_problem_dialog",
            None)
        selected_button = getattr(
            self,
            "source_validation_accept_selected_button",
            None)
        all_button = getattr(
            self,
            "source_validation_accept_all_button",
            None)
        if dialog is None or selected_button is None or all_button is None:
            return
        try:
            if not dialog.winfo_exists():
                return
        except tk.TclError:
            return
        callback_available = (
            self.source_validation_accept_callback is not None
            and not self.source_action_in_progress)
        selected = self._reviewable_validation_problems(
            selected_only=True)
        all_reviewable = self._reviewable_validation_problems(
            selected_only=False)
        selected_button.configure(
            state=(
                tk.NORMAL
                if callback_available and selected
                else tk.DISABLED))
        all_button.configure(
            state=(
                tk.NORMAL
                if callback_available and all_reviewable
                else tk.DISABLED))

    def _accept_selected_validation_problems(self):
        self._accept_validation_problems(
            self._reviewable_validation_problems(
                selected_only=True))

    def _accept_all_validation_problems(self):
        self._accept_validation_problems(
            self._reviewable_validation_problems(
                selected_only=False))

    def _accept_validation_problems(self, problems):
        problems = tuple(problems)
        if not problems:
            return
        context = getattr(
            self,
            "source_validation_problem_context",
            None)
        if not isinstance(context, dict):
            return
        remaining = int(
            context.get("report", {}).get(
                "remaining_problem_count",
                len(problems)))
        completes_chunk = len(problems) >= remaining
        consequence = (
            "Because this resolves every remaining issue, the chunk will be "
            "included and deck packaging/import may start immediately."
            if completes_chunk
            else (
                "Other issues will remain invalid and must still be reviewed "
                "or retried."
            ))
        if not messagebox.askyesno(
                "Accept reviewed content issue(s)?",
                (
                    f"Accept {len(problems):,} selected content validation "
                    f"issue{'' if len(problems) == 1 else 's'}?\n\n"
                    "This records a human override; it does not edit the "
                    "model response. JSON syntax, card shape, required fields, "
                    "and field types will still be enforced.\n\n"
                    f"{consequence}"
                ),
                parent=self.source_validation_problem_dialog):
            return
        reason = self.source_validation_reason.get().strip() or None
        record = context["record"]
        self._dispatch_source_action(
            "validation_accept",
            self.source_validation_accept_callback,
            {
                "job_id": record["job_id"],
                "problem_ids": tuple(
                    problem["problem_id"]
                    for problem in problems),
                "reason": reason,
            })

    @staticmethod
    def _source_job_is_retryable(record):
        status = record["status"].lower()
        return (
            status in {
                "cancelled",
                "connection_failed",
                "failed",
                "invalid",
                "invalid_response",
            }
            or (
                status == "pending"
                and not record["job_id"].endswith("::finalize")))

    def _retry_selected_source_jobs(self):
        selected = self._selected_source_jobs()
        failed = tuple(
            record
            for record in selected
            if self._source_job_is_retryable(record))
        if not failed:
            messagebox.showinfo(
                "No failed jobs selected",
                "Select one or more pending, failed, or invalid-response "
                "requests.",
                parent=self.root)
            return
        self._confirm_and_retry_source_jobs(failed)

    def _retry_all_source_jobs(self):
        failed = tuple(
            record
            for record in self.source_job_by_tree_id.values()
            if self._source_job_is_retryable(record))
        if not failed:
            messagebox.showinfo(
                "No failed jobs",
                "There are no saved incomplete requests to resume or retry.",
                parent=self.root)
            return
        self._confirm_and_retry_source_jobs(failed)

    def _confirm_and_retry_source_jobs(self, records):
        collect_only = all(
            record.get("execution_mode") == "economy"
            and record["status"].lower() == "pending"
            and "Economy Batch" in record.get("detail", "")
            for record in records)
        title = (
            "Collect Economy Batch?"
            if collect_only
            else "Authorize paid retries?")
        message = (
            (
                f"Check and collect {len(records):,} selected Economy Batch "
                "request(s)? This does not resubmit them. If OpenAI is still "
                "processing the batch, the rows will remain pending."
            )
            if collect_only
            else (
                f"Resume or retry {len(records):,} request(s)? Failed "
                "requests may incur additional OpenAI charges. Completed "
                "requests will not be repeated."
            ))
        if not messagebox.askyesno(
                title,
                message,
                parent=self.root):
            return
        self._dispatch_source_action(
            "retry",
            self.source_retry_callback,
            {
                "job_ids": tuple(
                    record["job_id"]
                    for record in records),
                "paid_confirmed": True,
            })

    def _generate_mode_changed(self, _event=None):
        source_selected = (
            hasattr(self, "generate_notebook")
            and self.generate_notebook.select()
            == str(self.from_source_tab))
        if source_selected:
            self._ensure_from_source_tab_built()
        self._refresh_card_setup_generation_context()
        if source_selected:
            self._schedule_source_estimate()

    def _refresh_card_setup_generation_context(self):
        editors = tuple(getattr(self, "pipeline_rows", ()))
        for editor in editors:
            editor.refresh_generation_language()
        if editors:
            self.schedule_pipeline_save()
            self._update_pipeline_status()

    def _source_page_changed(self, _event=None):
        selected = self.source_notebook.select()
        if selected == str(self.source_generate_page):
            self._schedule_source_estimate()
        elif selected == str(self.source_preview_page):
            try:
                option = self._selected_source_option()
            except ValueError:
                self.source_preview_status.set(
                    "Select a prepared source to inspect.")
            else:
                loaded_key = (
                    self.source_preview_page_data.source_key
                    if self.source_preview_page_data is not None
                    else None)
                if (
                        loaded_key != option.key
                        and not self.source_preview_pending):
                    self._request_source_preview(offset=0)
        elif selected == str(self.source_jobs_page):
            self._ensure_source_jobs_page_built()
            self._refresh_source_jobs(show_errors=False)

    def _build_pipeline_tab(self):
        if self._pipeline_tab_built:
            return
        self._pipeline_tab_built = True
        tab = self.pipeline_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        self.pipeline_scroll_frame = ScrollableFrame(
            tab,
            background=self.WINDOW_BACKGROUND,
            frame_style="App.TFrame")
        self.pipeline_scroll_frame.grid(
            row=0,
            column=0,
            sticky="nsew")
        page = self.pipeline_scroll_frame.content
        page.columnconfigure(0, weight=1)

        header_surface = RoundedPanel(
            page,
            fill=self.PALE,
            outline="#C8DDD8",
            background=self.WINDOW_BACKGROUND,
            radius=20,
            inset=14)
        header_surface.grid(row=0, column=0, sticky="ew")
        header = header_surface.interior
        header.columnconfigure(0, weight=1)

        tk.Label(
            header,
            text="CARD GENERATION",
            background=self.PALE,
            foreground=self.TEXT_PRIMARY,
            font=("DejaVu Sans", 11, "bold")).grid(
                row=0,
                column=0,
                sticky="w")
        tk.Label(
            header,
            text=(
                "Choose the language of your input, then select every card "
                "format you want to create."),
            background=self.PALE,
            foreground=self.TEXT_SECONDARY,
            font=("DejaVu Sans", 9),
            wraplength=880,
            justify="left").grid(
                row=1,
                column=0,
                sticky="w",
                pady=(4, 0))

        self.pipeline_editor_container = ttk.Frame(
            page,
            style="App.TFrame")
        self.pipeline_editor_container.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(10, 0))
        self.pipeline_editor_container.columnconfigure(0, weight=1)

        ttk.Label(
            page,
            textvariable=self.pipeline_status,
            style="Status.TLabel").grid(
                row=2,
                column=0,
                sticky="w",
                pady=(10, 0))
        self._render_pipeline_rows()

    def _ensure_pipeline_tab_built(self):
        if not getattr(self, "_pipeline_tab_built", False):
            self._build_pipeline_tab()

    def _build_advanced_tab(self):
        if self._advanced_tab_built:
            return
        self._advanced_tab_built = True
        tab = self.advanced_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        header_surface = RoundedPanel(
            tab,
            fill=self.PALE,
            outline="#C8DDD8",
            background=self.WINDOW_BACKGROUND,
            radius=20,
            inset=14)
        header_surface.grid(
            row=0,
            column=0,
            sticky="ew")
        header = header_surface.interior
        header.columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="PROMPT COMPONENT EDITOR",
            background=self.PALE,
            foreground=self.TEXT_PRIMARY,
            font=("DejaVu Sans", 11, "bold")).grid(
                row=0,
                column=0,
                sticky="w")
        tk.Label(
            header,
            text=(
                "Inspect the reusable instructions assembled for each "
                "request. Changes affect future paid requests, so editing "
                "is locked by default."),
            background=self.PALE,
            foreground=self.TEXT_SECONDARY,
            font=("DejaVu Sans", 9),
            wraplength=980,
            justify="left").grid(
                row=1,
                column=0,
                sticky="w",
                pady=(4, 0))

        editor_surface = RoundedPanel(
            tab,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.WINDOW_BACKGROUND,
            radius=26,
            inset=18)
        editor_surface.grid(
            row=1,
            column=0,
            sticky="nsew",
            pady=(10, 0))
        editor = editor_surface.interior
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(2, weight=1)

        self.prompt_options = pipeline_store.discover_prompts()
        self.prompt_options_by_key = {
            prompt.key: prompt
            for prompt in self.prompt_options
        }
        first_prompt_key = (
            self.gui_preferences["prompt_key"]
            if self.gui_preferences["prompt_key"]
            in self.prompt_options_by_key
            else (
                self.prompt_options[0].key
                if self.prompt_options
                else ""))
        self.prompt_selector_value = tk.StringVar(
            value=first_prompt_key)
        self.prompt_editing_enabled = tk.BooleanVar(value=False)
        self.prompt_status = tk.StringVar(
            value="Read-only · tick Enable prompt editing to make changes")
        self.current_prompt_key = None
        self.prompt_dirty = False
        self._prompt_loading = False
        self._prompt_selection_reverting = False

        toolbar = ttk.Frame(
            editor,
            style="Panel.TFrame")
        toolbar.grid(
            row=0,
            column=0,
            sticky="ew")
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(
            toolbar,
            text="COMPONENT",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                padx=(0, 8))
        self.prompt_selector = ttk.Combobox(
            toolbar,
            textvariable=self.prompt_selector_value,
            values=tuple(
                prompt.key
                for prompt in self.prompt_options),
            state="readonly",
            style="App.TCombobox")
        self.prompt_selector.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(0, 18))
        self.prompt_selector.bind(
            "<<ComboboxSelected>>",
            self._prompt_selection_changed,
            add="+")
        self.prompt_selector.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")
        self.prompt_editing_checkbox = ttk.Checkbutton(
            toolbar,
            text="Enable prompt editing",
            variable=self.prompt_editing_enabled,
            command=self._toggle_prompt_editing,
            style="Panel.TCheckbutton")
        self.prompt_editing_checkbox.grid(
            row=0,
            column=2,
            sticky="e")

        ttk.Label(
            editor,
            textvariable=self.prompt_status,
            style="Muted.TLabel").grid(
                row=1,
                column=0,
                sticky="w",
                pady=(8, 8))

        text_surface = RoundedPanel(
            editor,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.PANEL_BACKGROUND,
            radius=20,
            inset=5)
        text_surface.grid(
            row=2,
            column=0,
            sticky="nsew")
        text_editor = text_surface.interior
        text_editor.columnconfigure(0, weight=1)
        text_editor.rowconfigure(0, weight=1)
        self.prompt_text = tk.Text(
            text_editor,
            wrap=tk.WORD,
            undo=True,
            font=("DejaVu Sans Mono", 10),
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            insertbackground=self.TEXT_PRIMARY,
            selectbackground="#D7E9E5",
            selectforeground=self.TEXT_PRIMARY,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=12,
            pady=10,
            state=tk.DISABLED)
        prompt_scrollbar = RoundedScrollbar(
            text_editor,
            command=self.prompt_text.yview,
            background=self.PANEL_BACKGROUND,
            active=self.ACCENT)
        self.prompt_text.configure(
            yscrollcommand=prompt_scrollbar.set)
        self.prompt_text.grid(
            row=0,
            column=0,
            sticky="nsew")
        prompt_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns")
        self.prompt_text.bind(
            "<<Modified>>",
            self._prompt_text_modified,
            add="+")
        self.prompt_text.bind(
            "<FocusOut>",
            self._clear_text_selection,
            add="+")

        actions = ttk.Frame(
            editor,
            style="Panel.TFrame")
        actions.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        self.save_prompt_button = ttk.Button(
            actions,
            text="Save component",
            command=self.save_current_prompt,
            state=tk.DISABLED,
            style="Accent.TButton",
            cursor="hand2")
        self.save_prompt_button.grid(
            row=0,
            column=1,
            sticky="e")

        if first_prompt_key:
            self._load_prompt(first_prompt_key)
        else:
            self.prompt_status.set(
                "No files were found in input/prompt_components.")
            self.prompt_selector.configure(state=tk.DISABLED)
            self.prompt_editing_checkbox.configure(state=tk.DISABLED)
        self._install_advanced_preference_tracking()

    def _ensure_advanced_tab_built(self):
        if not getattr(self, "_advanced_tab_built", False):
            self._build_advanced_tab()

    def _build_help_tab(self):
        if self._help_tab_built:
            return
        self._help_tab_built = True
        tab = self.help_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        viewport = ScrollableFrame(
            tab,
            background=self.WINDOW_BACKGROUND,
            frame_style="App.TFrame")
        viewport.grid(row=0, column=0, sticky="nsew")
        content = viewport.content
        content.columnconfigure(0, weight=1)

        def section(row, title, body):
            ttk.Label(
                content,
                text=title,
                style="HelpSection.TLabel").grid(
                    row=row,
                    column=0,
                    sticky="w",
                    pady=(0 if row == 0 else 22, 5))
            ttk.Label(
                content,
                text=body,
                style="Status.TLabel",
                justify="left",
                wraplength=1160).grid(
                    row=row + 1,
                    column=0,
                    sticky="ew")

        section(
            0,
            "QUICK START",
            (
                "1. Install AnkiConnect in Anki: Tools → Add-ons → "
                "Get Add-ons, enter 2055492159, then restart Anki.\n"
                "2. In Card setup, configure each language's card "
                "directions, response fields and languages, and existing "
                "destination decks.\n"
                "3. In Generate, explicitly choose the input language, "
                "enter words or notes, and choose Generate and import. "
                "AutoAnki waits while Anki opens or syncs."))

        code_row = ttk.Frame(content, style="App.TFrame")
        code_row.grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Button(
            code_row,
            text="Copy add-on code 2055492159",
            command=self._copy_anki_addon_code,
            style="Secondary.TButton",
            cursor="hand2").grid(row=0, column=0, sticky="w")

        section(
            3,
            "USING ANKI ON ANOTHER COMPUTER",
            (
                "On the Anki computer, open Tools → Add-ons → AnkiConnect "
                "→ Config. Set webBindAddress to that computer's private LAN "
                "address (or 0.0.0.0), set a strong apiKey, restart Anki, and "
                "allow TCP port 8765 through the firewall only on a trusted "
                "LAN or VPN. Do not expose AnkiConnect directly to the "
                "internet.\n\n"
                "Below, enter http://ANKI-COMPUTER-IP:8765 and the same API "
                "key. AutoAnki uploads the generated package through "
                "AnkiConnect before importing it; the two computers do not "
                "need a shared filesystem. AutoAnki cannot start Anki on a "
                "remote computer, so Anki and its profile must already be "
                "open there."))

        connection_surface = RoundedPanel(
            content,
            fill=self.PANEL_BACKGROUND,
            outline=self.LINE,
            background=self.WINDOW_BACKGROUND,
            radius=24,
            inset=18)
        connection_surface.grid(
            row=5,
            column=0,
            sticky="ew",
            pady=(12, 0))
        connection = connection_surface.interior
        connection.columnconfigure(1, weight=1)

        ttk.Label(
            connection,
            text="ANKICONNECT ADDRESS",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                padx=(0, 12))
        self.anki_url_entry = ttk.Entry(
            connection,
            textvariable=self.anki_url)
        self.anki_url_entry.grid(
            row=0,
            column=1,
            sticky="ew")
        self.anki_url_entry.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")

        ttk.Label(
            connection,
            text="API KEY",
            style="FieldLabel.TLabel").grid(
                row=1,
                column=0,
                sticky="w",
                padx=(0, 12),
                pady=(12, 0))
        self.anki_api_key_entry = ttk.Entry(
            connection,
            textvariable=self.anki_api_key,
            show="•")
        self.anki_api_key_entry.grid(
            row=1,
            column=1,
            sticky="ew",
            pady=(12, 0))
        self.anki_api_key_entry.bind(
            "<FocusOut>",
            self._clear_entry_selection,
            add="+")

        action_row = ttk.Frame(connection, style="Panel.TFrame")
        action_row.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(14, 0))
        action_row.columnconfigure(0, weight=1)
        ttk.Label(
            action_row,
            textvariable=self.anki_connection_status,
            style="Muted.TLabel",
            wraplength=760).grid(
                row=0,
                column=0,
                sticky="w",
                padx=(0, 12))
        self.test_anki_connection_button = ttk.Button(
            action_row,
            text="Save and test",
            command=self.save_and_test_anki_connection,
            style="Accent.TButton",
            cursor="hand2")
        self.test_anki_connection_button.grid(
            row=0,
            column=1,
            sticky="e")

        section(
            6,
            "PORTABLE WINDOWS BUILD",
            (
                "A packaged AutoAnki.exe includes Python and its libraries, "
                "so the receiving computer does not need Python or Git. The "
                "Windows executable must be built on Windows. Settings, API "
                "keys, edited prompts, and generated packages are retained "
                "in the current Windows user's AppData\\Roaming\\AutoAnki "
                "folder rather than inside the executable."))
        viewport.bind_mousewheel_tree()

    def _ensure_help_tab_built(self):
        if not getattr(self, "_help_tab_built", False):
            self._build_help_tab()

    def _copy_anki_addon_code(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(
            anki_integration.ANKI_CONNECT_ADDON_CODE)
        self.anki_connection_status.set(
            "Copied AnkiConnect add-on code 2055492159")

    def save_and_test_anki_connection(self):
        try:
            settings = anki_integration.save_anki_connection_settings(
                self.anki_url.get(),
                self.anki_api_key.get())
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Could not save Anki connection",
                str(error),
                parent=self.root)
            return

        self.anki_url.set(settings.url)
        self.anki_connection_status.set(
            "Saved · testing AnkiConnect…")
        self.test_anki_connection_button.configure(
            state=tk.DISABLED,
            text="Testing…")
        worker = threading.Thread(
            target=self._test_anki_connection_in_background,
            args=(settings,),
            daemon=True)
        worker.start()
        self.root.after(
            self.POLL_INTERVAL_MS,
            self._poll_anki_connection_result)

    def _test_anki_connection_in_background(self, settings):
        try:
            client = anki_integration.AnkiConnectClient(
                url=settings.url,
                api_key=settings.api_key,
                timeout=5)
            deck_names = client.invoke("deckNames")
        except Exception as error:
            self.connection_result_queue.put(("error", error))
        else:
            self.connection_result_queue.put(
                ("success", tuple(sorted(deck_names))))

    def _poll_anki_connection_result(self):
        try:
            outcome, value = (
                self.connection_result_queue.get_nowait())
        except queue.Empty:
            self.root.after(
                self.POLL_INTERVAL_MS,
                self._poll_anki_connection_result)
            return

        self.test_anki_connection_button.configure(
            state=tk.NORMAL,
            text="Save and test")
        if outcome == "error":
            self.anki_connection_status.set(
                f"Saved, but connection failed · {value}")
            messagebox.showerror(
                "Anki connection failed",
                (
                    f"{value}\n\n"
                    "The settings were saved. Check the Help instructions, "
                    "then correct them and test again."),
                parent=self.root)
            return

        self.anki_connection_status.set(
            f"Connected · {len(value)} Anki decks available")
        self.deck_options = value
        for row in self.pipeline_rows:
            row.update_deck_options(value)
        try:
            pipeline_store.save_anki_deck_cache(value)
        except OSError:
            pass

    def _load_prompt(self, prompt_key):
        prompt = self.prompt_options_by_key[prompt_key]
        text = prompt.path.read_text(encoding="utf-8")
        self._prompt_loading = True
        try:
            self.prompt_text.configure(state=tk.NORMAL)
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", text)
            self.prompt_text.edit_reset()
            self.prompt_text.edit_modified(False)
            self.prompt_text.configure(
                state=(
                    tk.NORMAL
                    if self.prompt_editing_enabled.get()
                    else tk.DISABLED),
                foreground=(
                    self.TEXT_PRIMARY
                    if self.prompt_editing_enabled.get()
                    else self.TEXT_SECONDARY))
        finally:
            self._prompt_loading = False
        self.current_prompt_key = prompt_key
        self._refresh_prompt_selector_display()
        self.prompt_dirty = False
        self._update_prompt_editor_state()

    def _refresh_prompt_selector_display(self):
        """Make the readonly combobox display its concrete selected item."""
        prompt_key = self.current_prompt_key
        if not prompt_key:
            return
        try:
            index = next(
                index
                for index, prompt in enumerate(self.prompt_options)
                if prompt.key == prompt_key)
            self.prompt_selector.current(index)
        except (StopIteration, tk.TclError):
            self.prompt_selector_value.set(prompt_key)

    def _update_prompt_editor_state(self, message=None):
        editing = self.prompt_editing_enabled.get()
        if message is None:
            if self.prompt_dirty:
                message = "Editing enabled · unsaved changes"
            elif editing:
                message = "Editing enabled · no unsaved changes"
            else:
                message = (
                    "Read-only · tick Enable prompt editing to make changes")
        self.prompt_status.set(message)
        self.save_prompt_button.configure(
            state=(
                tk.NORMAL
                if editing and self.prompt_dirty
                else tk.DISABLED))

    def _prompt_text_modified(self, _event=None):
        if self._prompt_loading:
            return
        try:
            modified = self.prompt_text.edit_modified()
        except tk.TclError:
            return
        if not modified:
            return
        self.prompt_text.edit_modified(False)
        if self.prompt_editing_enabled.get():
            self.prompt_dirty = True
            self._update_prompt_editor_state()

    def _confirm_unsaved_prompt(self, action):
        if not self.prompt_dirty:
            return "discard"
        answer = messagebox.askyesnocancel(
            "Unsaved prompt changes",
            (
                f"Save changes to {self.current_prompt_key} before "
                f"{action}?\n\n"
                "Yes saves, No discards, and Cancel keeps the editor open."),
            parent=self.root)
        if answer is None:
            return "cancel"
        return "save" if answer else "discard"

    def _discard_current_prompt_changes(self):
        if self.current_prompt_key is not None:
            self._load_prompt(self.current_prompt_key)

    def _toggle_prompt_editing(self):
        if self.prompt_editing_enabled.get():
            self.prompt_text.configure(
                state=tk.NORMAL,
                foreground=self.TEXT_PRIMARY)
            self.prompt_text.focus_set()
            self._update_prompt_editor_state()
            return

        decision = self._confirm_unsaved_prompt("disabling editing")
        if decision == "cancel":
            self.prompt_editing_enabled.set(True)
            self._update_prompt_editor_state()
            return
        if decision == "save":
            if not self.save_current_prompt(
                    show_confirmation=False,
                    require_editing=False):
                self.prompt_editing_enabled.set(True)
                self._update_prompt_editor_state()
                return
        elif self.prompt_dirty:
            self._discard_current_prompt_changes()

        self.prompt_text.configure(
            state=tk.DISABLED,
            foreground=self.TEXT_SECONDARY)
        self._update_prompt_editor_state()

    def _prompt_selection_changed(self, _event=None):
        if self._prompt_selection_reverting:
            return
        prompt_key = self.prompt_selector_value.get()
        if (
                not prompt_key
                or prompt_key == self.current_prompt_key):
            return

        decision = self._confirm_unsaved_prompt("switching prompts")
        if decision == "cancel":
            self._prompt_selection_reverting = True
            try:
                self.prompt_selector_value.set(self.current_prompt_key)
            finally:
                self._prompt_selection_reverting = False
            return
        if decision == "save":
            if not self.save_current_prompt(show_confirmation=False):
                self._prompt_selection_reverting = True
                try:
                    self.prompt_selector_value.set(self.current_prompt_key)
                finally:
                    self._prompt_selection_reverting = False
                return

        try:
            self._load_prompt(prompt_key)
        except (OSError, UnicodeError) as error:
            messagebox.showerror(
                "Could not load prompt",
                str(error),
                parent=self.root)
            self._prompt_selection_reverting = True
            try:
                self.prompt_selector_value.set(self.current_prompt_key)
            finally:
                self._prompt_selection_reverting = False

    def save_current_prompt(
            self,
            show_confirmation=True,
            require_editing=True):
        if (
                (
                    require_editing
                    and not self.prompt_editing_enabled.get())
                or self.current_prompt_key is None):
            return False
        text = self.prompt_text.get("1.0", "end-1c")
        prompt = self.prompt_options_by_key[self.current_prompt_key]
        try:
            pipeline_store.save_prompt_text(
                prompt.path,
                text)
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Could not save prompt",
                str(error),
                parent=self.root)
            return False

        self.prompt_dirty = False
        self.prompt_text.edit_modified(False)
        self._update_prompt_editor_state(
            f"Saved {self.current_prompt_key}")
        if hasattr(self, "source_estimate_price"):
            # Source prompts are part of both the cost and immutable paid
            # request contract. A saved edit therefore needs a new estimate
            # and a fresh authorization.
            self._schedule_source_estimate()
        if show_confirmation:
            messagebox.showinfo(
                "Prompt saved",
                f'Saved "{self.current_prompt_key}".',
                parent=self.root)
        return True

    def _render_pipeline_rows(self):
        for row in self.pipeline_rows:
            row.destroy()
        config = (
            self.pipeline_configs[0]
            if self.pipeline_configs
            else pipeline_store.default_pipeline())
        self.pipeline_rows = [PipelineEditor(
            self,
            self.pipeline_editor_container,
            config)]
        self.pipeline_scroll_frame.bind_mousewheel_tree()
        for row in self.pipeline_rows:
            row.bind_target_deck_mousewheel()
        self._update_pipeline_status()

    def _update_pipeline_status(self, message=None):
        if message is None:
            try:
                config = self.pipeline_rows[0].to_config()
                language = pipeline_store.get_language(
                    config.language_key)
                output_count = len(
                    pipeline_store.get_enabled_cards(config))
                if output_count:
                    output_word = (
                        "output" if output_count == 1 else "outputs")
                    deck_mode = (
                        "separate target decks"
                        if config.separate_target_decks
                        else "one target deck")
                    message = (
                        f"Generation language: {language.name} · "
                        f"{output_count} card "
                        f"{output_word} selected · {deck_mode} · "
                        f"{len(pipeline_store.get_requested_field_settings(config))} "
                        "unique response fields · "
                        "1 paid API request per run")
                else:
                    message = (
                        f"Generation language: {language.name} · "
                        "Select at least one card output.")
            except (IndexError, ValueError):
                message = (
                    "Select a language and at least one card output.")
        self.pipeline_status.set(message)

    def _show_pipeline_notice(self, message):
        if self.pipeline_notice_after_id is not None:
            self.root.after_cancel(self.pipeline_notice_after_id)
        self.pipeline_status.set(message)
        self.pipeline_notice_after_id = self.root.after(
            2500,
            self._clear_pipeline_notice)

    def _clear_pipeline_notice(self):
        self.pipeline_notice_after_id = None
        self._update_pipeline_status()

    def get_pipeline_configs(self):
        pipelines = (
            tuple(
                row.to_config()
                for row in self.pipeline_rows)
            if self._pipeline_tab_built
            else tuple(self.pipeline_configs))
        pipelines = pipeline_store.validate_pipelines(pipelines)

        for pipeline in pipelines:
            prompt_builder.build_prompt(pipeline)
        return pipelines

    def schedule_pipeline_save(self, *_args):
        if self.pipeline_save_after_id is not None:
            self.root.after_cancel(self.pipeline_save_after_id)
        self.pipeline_save_after_id = self.root.after(
            400,
            self._autosave_pipeline_rows)
        if hasattr(self, "source_estimate_price"):
            self._sync_source_example_control()
            self._schedule_source_estimate()

    def _autosave_pipeline_rows(self):
        self.pipeline_save_after_id = None
        self.save_pipeline_rows(show_errors=False)

    def save_pipeline_rows(
            self,
            *,
            show_confirmation=False,
            show_errors=True):
        try:
            pipelines = self.get_pipeline_configs()
            self.pipeline_saver(pipelines)
        except (OSError, ValueError) as error:
            if show_errors:
                messagebox.showerror(
                    "Could not save pipelines",
                    str(error),
                    parent=self.root)
            else:
                self.pipeline_status.set(
                    f"Not saved yet · {error}")
            return None

        self.pipeline_configs = pipelines
        for row, pipeline in zip(
                self.pipeline_rows,
                pipelines):
            row.config = pipeline
        self._show_pipeline_notice(
            "Card settings saved automatically.")
        if show_confirmation:
            messagebox.showinfo(
                "Pipelines saved",
                "Your pipeline selections have been saved.",
                parent=self.root)
        return pipelines

    def _main_tab_changed(self, _event=None):
        selected_tab = self.notebook.select()
        if selected_tab == str(self.advanced_tab):
            self._ensure_advanced_tab_built()
            self._refresh_prompt_selector_display()
            self.root.after_idle(
                self._refresh_prompt_selector_display)
            return
        if selected_tab == str(self.help_tab):
            self._ensure_help_tab_built()
            return
        if selected_tab == str(self.generate_tab):
            if (
                    hasattr(self, "generate_notebook")
                    and self.generate_notebook.select()
                    == str(self.from_source_tab)):
                self._ensure_from_source_tab_built()
                self._schedule_source_estimate()
            return
        if selected_tab != str(self.pipeline_tab):
            return
        self._ensure_pipeline_tab_built()
        self.deck_status.set(
            "Connecting to Anki and refreshing decks…")
        self.refresh_anki_decks(launch_if_needed=True)

    def get_generation_language(self):
        selected_name = self.generation_language.get()
        for language in pipeline_store.list_languages():
            if language.name == selected_name:
                return language
        raise ValueError("Select an input language on the Generate tab.")

    def get_card_setup_generation_language(self):
        """Use the prepared source's language while From Source is active."""
        if (
                hasattr(self, "generate_notebook")
                and hasattr(self, "from_source_tab")
                and self.generate_notebook.select()
                == str(self.from_source_tab)
                and hasattr(self, "source_language_label")):
            selected_name = self.source_language_label.get().strip()
            for language in pipeline_store.list_languages():
                if language.name == selected_name:
                    return language
        return self.get_generation_language()

    def _generation_language_changed(self, _event=None):
        self._ensure_pipeline_tab_built()
        language = self.get_generation_language()
        for editor in getattr(self, "pipeline_rows", ()):
            editor.refresh_generation_language()
        self._sync_learned_filter_controls()
        self.schedule_pipeline_save()
        self._update_pipeline_status()
        self._set_status(
            f"{language.name} selected for the next generation.",
            self.TEXT_SECONDARY)

    def _deck_selector_opened(self, _event=None):
        self.deck_status.set(
            "Refreshing Anki decks…")
        self.refresh_anki_decks(launch_if_needed=True)

    def refresh_anki_decks(self, launch_if_needed=False):
        if self.deck_refresh_in_progress:
            self.deck_launch_requested = (
                self.deck_launch_requested
                or launch_if_needed)
            return
        self.deck_refresh_in_progress = True
        worker = threading.Thread(
            target=self._load_anki_decks_in_background,
            args=(launch_if_needed,),
            daemon=True)
        worker.start()
        self.root.after(
            self.POLL_INTERVAL_MS,
            self._poll_deck_result)

    def _load_anki_decks_in_background(
            self,
            launch_if_needed=False):
        try:
            client = anki_integration.AnkiConnectClient(timeout=3)
            if launch_if_needed:
                anki_integration.ensure_anki_running(client)
                deck_names = (
                    anki_integration.wait_for_collection_ready(
                        client))
            elif not client.is_available(raise_response_errors=True):
                self.deck_result_queue.put(("unavailable", None))
                return
            else:
                deck_names = client.invoke("deckNames")
        except Exception as error:
            self.deck_result_queue.put(("error", error))
        else:
            self.deck_result_queue.put(
                ("success", tuple(sorted(deck_names))))

    def _poll_deck_result(self):
        try:
            outcome, value = (
                self.deck_result_queue.get_nowait())
        except queue.Empty:
            self.root.after(
                self.POLL_INTERVAL_MS,
                self._poll_deck_result)
            return

        self.deck_refresh_in_progress = False
        if outcome == "success":
            try:
                pipeline_store.save_anki_deck_cache(value)
            except OSError:
                pass
            if value != self.deck_options:
                self.deck_options = value
                for row in self.pipeline_rows:
                    row.update_deck_options(value)
                for learned_row in getattr(
                        self,
                        "learned_filter_rows",
                        ()):
                    try:
                        learned_row["deck_box"].configure(values=value)
                    except tk.TclError:
                        pass
                self._update_pipeline_status()
            if hasattr(self, "deck_status"):
                self.deck_status.set(
                    f"{len(value)} Anki decks available · updates automatically")
        elif outcome == "unavailable":
            if hasattr(self, "deck_status"):
                self.deck_status.set(
                    "Anki is closed · showing saved decks; type a deck "
                    "name or open Card setup to reconnect")
        elif hasattr(self, "deck_status"):
            self.deck_status.set(
                f"Could not refresh Anki decks · {value}")

        if getattr(self, "deck_launch_requested", False):
            self.deck_launch_requested = False
            self.refresh_anki_decks(launch_if_needed=True)
            return

        self.root.after(
            self.DECK_REFRESH_INTERVAL_MS,
            self.refresh_anki_decks)

    def _update_input_count(self, _event=None):
        value = self.input_text.get("1.0", "end-1c")
        line_count = len(value.splitlines()) if value else 0
        self.input_count.set(
            f"{line_count} lines · {len(value)} characters")

    def _restore_manual_input_draft(self):
        draft = self.gui_preferences.get("manual_input_draft", "")
        if draft:
            self.input_text.insert("1.0", draft)
        self.input_text.edit_modified(False)

    def _manual_input_changed(self, _event=None):
        if not self.input_text.edit_modified():
            return
        self.input_text.edit_modified(False)
        self._update_input_count()
        self._schedule_preferences_save()

    def _set_status(self, message, colour=None):
        self.status.set(message)
        if colour is not None and hasattr(self, "status_dot"):
            self.status_dot.itemconfigure(
                self.status_circle,
                fill=colour,
                outline=colour)

    def _set_generation_busy(self, busy):
        self.generation_in_progress = busy
        self.generate_button.configure(
            state=tk.DISABLED if busy else tk.NORMAL,
            text=(
                "Generating and importing…"
                if busy
                else "Generate and import"))
        if hasattr(self, "api_key_button"):
            self.api_key_button.configure(
                state=tk.DISABLED if busy else tk.NORMAL)
        if hasattr(self, "generation_progress"):
            if busy:
                self.generation_progress.grid()
                self.generation_progress.start(12)
            else:
                self.generation_progress.stop()
                self.generation_progress.grid_remove()

    def close(self):
        if self.generation_in_progress:
            messagebox.showinfo(
                "Generation is still running",
                (
                    "Keep AutoAnki open until generation and import finish. "
                    "Closing now could discard a paid response before it is "
                    "imported."),
                parent=self.root)
            return
        if getattr(self, "source_action_in_progress", False):
            messagebox.showinfo(
                "Source work is still running",
                (
                    "Keep AutoAnki open while the active source coordinator "
                    "finishes. Per-chunk results already saved by the backend "
                    "remain recoverable from Jobs & Failures."
                ),
                parent=self.root)
            return
        if getattr(self, "prompt_dirty", False):
            decision = self._confirm_unsaved_prompt("closing AutoAnki")
            if decision == "cancel":
                return
            if (
                    decision == "save"
                    and not self.save_current_prompt(
                        show_confirmation=False,
                        require_editing=False)):
                return
        if self.pipeline_save_after_id is not None:
            self.root.after_cancel(self.pipeline_save_after_id)
            self.pipeline_save_after_id = None
        if self.pipeline_notice_after_id is not None:
            self.root.after_cancel(self.pipeline_notice_after_id)
            self.pipeline_notice_after_id = None
        if self.source_estimate_after_id is not None:
            self.root.after_cancel(self.source_estimate_after_id)
            self.source_estimate_after_id = None
        if self.preference_save_after_id is not None:
            self.root.after_cancel(self.preference_save_after_id)
            self.preference_save_after_id = None
        if self.save_pipeline_rows(show_errors=True) is None:
            return
        self._save_gui_preferences()
        self.root.destroy()

    def start_generation(self):
        words = self.input_text.get("1.0", tk.END).strip()
        if not words:
            messagebox.showwarning(
                "Nothing to generate",
                "Enter at least one word or note before generating cards.")
            return

        pipelines = self.save_pipeline_rows(
            show_errors=True)
        if pipelines is None:
            return
        if not pipelines:
            messagebox.showwarning(
                "No pipelines configured",
                "Add at least one pipeline before generating cards.")
            return

        generation_language = self.get_generation_language()
        learned_filter_enabled = self._language_filter(
            generation_language.key).enabled
        active_settings = (
            pipeline_store.get_language_settings(
                pipelines[0],
                generation_language.key)
            if isinstance(pipelines[0], pipeline_store.PipelineConfig)
            else None)
        make_character_items = bool(
            active_settings is not None
            and active_settings.make_items_for_characters
            and generation_language.model_language_key
            == "classical_chinese")
        if learned_filter_enabled or make_character_items:
            if self.manual_input_filter_callback is None:
                messagebox.showerror(
                    "Vocabulary preflight unavailable",
                    (
                        "This build has not connected the manual vocabulary "
                        "preflight. Disable the learned-word and character-"
                        "item options or reconnect the backend."
                    ),
                    parent=self.root)
                return
            try:
                filter_request = self._manual_input_filter_request(
                    words,
                    make_items_for_characters=make_character_items)
            except ValueError as error:
                messagebox.showerror(
                    "Learned-word filter is incomplete",
                    str(error),
                    parent=self.root)
                return
            self._set_generation_busy(True)
            self._set_status(
                "Preparing vocabulary items before generation…",
                self.WARNING)
            worker = threading.Thread(
                target=self._filter_manual_input_in_background,
                args=(filter_request, pipelines),
                daemon=True)
            worker.start()
            self.root.after(
                self.POLL_INTERVAL_MS,
                self._poll_result)
            return
        self._start_pipeline_generation(words, pipelines)

    def _start_pipeline_generation(
            self,
            words,
            pipelines,
            *,
            status_message=None):
        if not self._ensure_api_key():
            self._set_generation_busy(False)
            return
        self._set_generation_busy(True)
        self._set_status(
            status_message
            or f"Starting {len(pipelines)} generation pipelines…",
            self.WARNING)
        worker = threading.Thread(
            target=self._generate_in_background,
            args=(words, pipelines),
            daemon=True)
        worker.start()
        self.root.after(
            self.POLL_INTERVAL_MS,
            self._poll_result)

    def _filter_manual_input_in_background(self, request, pipelines):
        try:
            value = self.manual_input_filter_callback(request)
            result = normalise_manual_input_filter_response(
                value,
                len(request["candidates"]))
        except Exception as error:
            self.result_queue.put(
                ("manual_filter_error", error))
            return
        self.result_queue.put(
            ("manual_filter_complete", (result, pipelines)))

    def _ensure_api_key(self):
        try:
            api_key = process_text.get_api_key()
        except OSError as error:
            messagebox.showerror(
                "Could not read API key",
                str(error),
                parent=self.root)
            return False

        if api_key:
            return True
        return self.configure_api_key(
            show_confirmation=False)

    def configure_api_key(self, show_confirmation=True):
        api_key = simpledialog.askstring(
            "Configure OpenAI API key",
            (
                "Enter your OpenAI API key. It will be stored outside the "
                "repository with access restricted to your user account."),
            show="*",
            parent=self.root)
        if api_key is None:
            return False

        try:
            credential_store.save_api_key(api_key)
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Could not save API key",
                str(error),
                parent=self.root)
            return False

        self._set_status(
            "API key saved on this device.",
            self.ACCENT)
        if show_confirmation:
            messagebox.showinfo(
                "API key saved",
                (
                    "The key was saved outside the AutoAnki repository and "
                    "will be used for future generations."),
                parent=self.root)
        return True

    def forget_api_key(self):
        try:
            saved_api_key = credential_store.load_api_key()
        except OSError as error:
            messagebox.showerror(
                "Could not read API key",
                str(error),
                parent=self.root)
            return

        if not saved_api_key:
            messagebox.showinfo(
                "No saved API key",
                "AutoAnki does not have a saved API key to remove.",
                parent=self.root)
            return

        confirmed = messagebox.askyesno(
            "Forget saved API key",
            "Remove the API key saved by AutoAnki from this device?",
            parent=self.root)
        if not confirmed:
            return

        try:
            credential_store.delete_api_key()
        except OSError as error:
            messagebox.showerror(
                "Could not remove API key",
                str(error),
                parent=self.root)
            return

        if process_text.get_api_key():
            self._set_status(
                "Saved API key removed. OPENAI_API_KEY is still active.",
                self.TEXT_SECONDARY)
        else:
            self._set_status(
                "Saved API key removed.",
                self.TEXT_SECONDARY)

    def _generate_in_background(self, words, pipelines):
        try:
            execution_arguments = {
                "progress_callback": lambda progress: (
                    self.result_queue.put(
                        ("pipeline_progress", progress))),
            }
            if self.paid_dispatch_control is not None:
                execution_arguments["paid_dispatch_control"] = (
                    self.paid_dispatch_control)
            summary = self.pipeline_executor(
                words,
                pipelines,
                **execution_arguments)
        except Exception as error:
            self.result_queue.put(("run_error", error))
            return

        self.result_queue.put(("run_complete", summary))

    def _poll_result(self):
        try:
            outcome, value = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(
                self.POLL_INTERVAL_MS,
                self._poll_result)
            return

        if outcome == "manual_filter_error":
            self._set_generation_busy(False)
            self._set_status(
                "Learned-word check failed before generation.",
                self.ERROR)
            messagebox.showerror(
                "Could not check learned words",
                (
                    f"{value}\n\n"
                    "No OpenAI request was made and no pipeline was started."
                ),
                parent=self.root)
            return

        if outcome == "manual_filter_complete":
            result, pipelines = value
            if result.remaining_count == 0:
                self._set_generation_busy(False)
                self._set_status(
                    "Every input line is already present in the selected "
                    "Anki field.",
                    self.ACCENT)
                messagebox.showinfo(
                    "Nothing new to generate",
                    (
                        f"All {result.excluded_count:,} nonblank input lines "
                        "are already present. No OpenAI request was made and "
                        "no deck was created."
                    ),
                    parent=self.root)
                return
            self._start_pipeline_generation(
                result.filtered_text,
                pipelines,
                status_message=(
                    f"Prepared {result.remaining_count:,} vocabulary items"
                    + (
                        f" ({result.added_character_count:,} character "
                        "items added)"
                        if result.added_character_count
                        else "")
                    + (
                        f"; omitted {result.excluded_count:,} learned lines"
                        if result.excluded_count
                        else "")
                    + f"; starting {len(pipelines)} generation pipelines…"
                ))
            return

        if outcome == "pipeline_progress":
            action = (
                "requesting an OpenAI response"
                if value.stage == "generating"
                else "importing into Anki")
            self._set_status(
                f"Pipeline {value.index}/{value.total}: {action}…",
                self.WARNING)
            self.root.after(
                self.POLL_INTERVAL_MS,
                self._poll_result)
            return

        self._set_generation_busy(False)
        if outcome == "run_error":
            self._set_status(
                "Pipeline run failed.",
                self.ERROR)
            retry = messagebox.askretrycancel(
                "Generation failed",
                (
                    f"{value}\n\n"
                    "No retry has been made. Retrying may make another "
                    "paid OpenAI API request."),
                parent=self.root)
            if retry:
                self.start_generation()
            return

        self._show_run_summary(value)

    def _show_run_summary(self, summary):
        success_count = len(summary.successes)
        failure_count = len(summary.failures)

        if summary.successes:
            last_output = summary.successes[-1].output_path.resolve()
            if success_count == 1:
                self.output.set(str(last_output))
            else:
                self.output.set(
                    f"{success_count} packages generated under "
                    f"{process_text.OUTPUT_DIRECTORY}")

        if summary.failures:
            self._set_status(
                f"{success_count} pipelines succeeded; "
                f"{failure_count} failed.",
                self.ERROR)
            failure_lines = []
            for failure in summary.failures:
                line = (
                    f"• {failure.pipeline.generated_deck_name} "
                    f"({failure.stage}): {failure.error}")
                if failure.output_path:
                    line += (
                        f"\n  Package retained at "
                        f"{failure.output_path.resolve()}")
                failure_lines.append(line)
            failure_message = "\n\n".join(failure_lines)
            generation_failed = any(
                failure.stage == "generation"
                for failure in summary.failures)
            if generation_failed:
                retry = messagebox.askretrycancel(
                    "Generation failed",
                    (
                        f"{failure_message}\n\n"
                        "No retry has been made. Retrying may make another "
                        "paid OpenAI API request."),
                    parent=self.root)
                if retry:
                    self.start_generation()
            else:
                messagebox.showerror(
                    "Some pipelines failed",
                    failure_message,
                    parent=self.root)
            return

        self._set_status(
            f"Imported {summary.cards_moved} cards through "
            f"{success_count} pipelines.",
            self.ACCENT)
        messagebox.showinfo(
            "All pipelines complete",
            (
                f"Successfully ran {success_count} pipelines and imported "
                f"{summary.cards_moved} cards."),
            parent=self.root)


def main():
    import source_workflow

    root = tk.Tk()
    workflow = source_workflow.SourceWorkflowController()
    app = AutoAnkiApp(
        root,
        **workflow.gui_hooks())
    # Keep the coordinator explicitly reachable for the lifetime of Tk.
    app.source_workflow_controller = workflow
    root.mainloop()


if __name__ == "__main__":
    main()
