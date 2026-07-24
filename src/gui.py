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
import learned_filter_store
import pipeline_runner
import pipeline_store
import prompt_builder
import process_text


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


@dataclass(frozen=True)
class SourceUiOption:
    """One prepared source presented by the Generate UI."""

    key: str
    name: str
    word_count: int | None = None
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


BUILT_IN_SOURCE_OPTIONS = (
    SourceUiOption(
        key="daodejing_huijiao",
        name="Daodejing",
        word_count=922,
        section_count=81,
        source_language_key="classical_chinese_warring_states",
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
    "Classical Chinese (Warring States)",
    "Classical Chinese (Ming)",
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
    for name, number in (
            ("excluded_count", excluded_count),
            ("remaining_count", remaining_count)):
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
    if excluded_count + remaining_count != candidate_count:
        raise ValueError(
            "Manual-input filter counts do not match the requested lines.")
    return ManualInputFilterResult(
        filtered_text="\n".join(filtered_candidates),
        excluded_count=excluded_count,
        remaining_count=remaining_count)


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
    if low is not None or high is not None:
        low = float(low if low is not None else high)
        high = float(high if high is not None else low)
        price = f"A${low:,.2f}–A${high:,.2f}"
    elif exact is not None:
        price = f"A${float(exact):,.2f}"
    else:
        price = "Estimate unavailable"

    request_count = read("request_count")
    candidate_count = read("candidate_count")
    input_tokens = read("input_tokens")
    output_tokens = read("output_tokens")
    details = []
    if candidate_count is not None:
        details.append(f"{int(candidate_count):,} new words")
    if request_count is not None:
        details.append(f"{int(request_count):,} requests")
    if input_tokens is not None:
        details.append(f"{int(input_tokens):,} estimated input tokens")
    if output_tokens is not None:
        details.append(f"{int(output_tokens):,} estimated output tokens")
    largest_input = read("largest_request_input_tokens")
    largest_output = read("largest_request_output_tokens")
    if largest_input is not None and largest_output is not None:
        details.append(
            "largest request ≈ "
            f"{int(largest_input):,} input / "
            f"{int(largest_output):,} output tokens")
    if details:
        details.append("automatic transient retries are not included")
    assumptions = read("assumptions", {})
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

    def bind_mousewheel_tree(self):
        """Route wheel events from every child control to this viewport."""
        stack = [self.canvas, self.content]
        while stack:
            widget = stack.pop()
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
        self.direction_enabled_variables = {}
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
        self._update_all_visibility()

    def _trace(self, variable, callback=None):
        trace_id = variable.trace_add(
            "write",
            callback or self.app.schedule_pipeline_save)
        self.variable_traces.append((variable, trace_id))

    def _language_values(self, source_language_key, field_key):
        return tuple(
            language.name
            for language in pipeline_store.list_response_languages()
            if not (
                field_key == "translation"
                and language.key == source_language_key))

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
            target_values = self._language_values(
                language.key,
                field.key)
            selected_target_key = selected_by_key.get(
                field.key,
                remembered_by_key.get(field.key))
            if selected_target_key == language.key and (
                    field.key == "translation"):
                selected_target_key = None
            if selected_target_key is None:
                selected_target_key = (
                    "english"
                    if language.key != "english" or field.key != "translation"
                    else "french")
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

    def _add_language_tab(self, language, config):
        tab = ttk.Frame(
            self.language_notebook,
            style="Panel.TFrame")
        tab.columnconfigure(0, weight=1)
        self.language_notebook.add(tab, text=language.name)
        self.language_tabs[language.key] = tab
        self.language_by_tab[str(tab)] = language

        content = ttk.Frame(
            tab,
            padding=(20, 18, 20, 32),
            style="Panel.TFrame")
        content.grid(row=0, column=0, sticky="ew")
        content.columnconfigure(0, weight=1)
        settings = pipeline_store.get_language_settings(
            config,
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
                    row=1,
                    column=0,
                    columnspan=3,
                    sticky="w",
                    pady=(5, 6))
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
                row=2,
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
            self._trace(deck_variable)

        self.direction_enabled_variables[
            language.key] = enabled_variables
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
        self._update_field_box_states(language_key)
        self._refresh_layout_geometry(language_key)

    def _refresh_layout_geometry(self, language_key):
        """Resize rounded surfaces after their optional controls change."""
        if not self.frame.winfo_exists():
            return
        self.frame.update_idletasks()
        for surface in self.direction_surfaces[
                language_key].values():
            surface._fit_to_contents()
        self.shared_field_containers[
            language_key]._fit_to_contents()
        self.frame.update_idletasks()
        self.frame._fit_to_contents()
        self.frame.update_idletasks()
        scroll_frame = getattr(
            self.app,
            "pipeline_scroll_frame",
            None)
        if scroll_frame is not None:
            scroll_frame._content_changed()

    def _update_all_visibility(self):
        for language in pipeline_store.list_settings_languages():
            self._update_visibility(language.key)

    def _language_changed(self, _event=None):
        language = self.get_active_language()
        if language.key == self.last_active_language_key:
            return
        self.last_active_language_key = language.key
        self._update_visibility(language.key)

    def get_active_language(self):
        selected_tab = self.language_notebook.select()
        language = self.language_by_tab.get(selected_tab)
        if language is None:
            raise ValueError("Select a language.")
        return language

    def to_config(self):
        active_language = self.app.get_generation_language()
        all_settings = []
        for language in pipeline_store.list_settings_languages():
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
                                language.key][direction.key]))))
            all_settings.append(pipeline_store.LanguageSettings(
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
                shared_fields=shared_fields,
                shared_field_languages=shared_field_languages))
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
            source_retry_callback=None,
            source_inspect_callback=None,
            source_anki_options_loader=None,
            source_preview_loader=None,
            manual_input_filter_callback=None,
            learned_filter_loader=None,
            learned_filter_saver=None):
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
        self.source_retry_callback = source_retry_callback
        self.source_inspect_callback = source_inspect_callback
        self.source_anki_options_loader = source_anki_options_loader
        self.source_preview_loader = source_preview_loader
        self.manual_input_filter_callback = manual_input_filter_callback
        self.learned_filter_loader = (
            learned_filter_loader
            or learned_filter_store.load_language_filters)
        self.learned_filter_saver = (
            learned_filter_saver
            or learned_filter_store.save_language_filters)
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

        root.title("AutoAnki")
        root.configure(background=self.WINDOW_BACKGROUND)

        self.status = tk.StringVar(value="Ready to generate")
        self.output = tk.StringVar(
            value=str(process_text.DECK_PATH))
        self.input_count = tk.StringVar(
            value="0 lines · 0 characters")
        self.pipeline_status = tk.StringVar()
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
        self._build_widgets()
        self._render_pipeline_rows()
        self._set_initial_window_size()
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

        if self.pipeline_load_error:
            root.after(0, self._show_pipeline_load_error)

    def _show_pipeline_load_error(self):
        messagebox.showerror(
            "Could not load pipeline settings",
            (
                f"{self.pipeline_load_error}\n\n"
                "The default pipeline was loaded instead."),
            parent=self.root)

    def _set_initial_window_size(self):
        self.root.update_idletasks()

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = min(1860, screen_width - 80)
        height = min(1200, screen_height - 100)
        x_position = max((screen_width - width) // 2, 0)
        y_position = max((screen_height - height) // 2, 0)

        self.root.geometry(
            f"{width}x{height}+{x_position}+{y_position}")
        self.root.minsize(min(1000, width), min(720, height))

    def _configure_styles(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self._style_images = []

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

            def rgb(colour):
                value = colour.lstrip("#")
                return tuple(
                    int(value[index:index + 2], 16)
                    for index in (0, 2, 4))

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

            fill_rgb = rgb(fill)
            outline_rgb = rgb(outline or fill)
            backdrop_rgb = rgb(backdrop)
            samples = 4
            sample_count = samples * samples
            for y in range(size):
                for x in range(size):
                    totals = [0, 0, 0]
                    for sample_y in range(samples):
                        point_y = y + (sample_y + 0.5) / samples
                        for sample_x in range(samples):
                            point_x = x + (
                                sample_x + 0.5) / samples
                            if inside(point_x, point_y, 1):
                                colour = fill_rgb
                            elif inside(point_x, point_y):
                                colour = outline_rgb
                            else:
                                colour = backdrop_rgb
                            for channel in range(3):
                                totals[channel] += colour[channel]
                    colour = tuple(
                        round(total / sample_count)
                        for total in totals)
                    image.put(
                        "#%02x%02x%02x" % colour,
                        (x, y))
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

            fill_rgb = colour_rgb(fill)
            outline_rgb = colour_rgb(outline)
            check_rgb = colour_rgb(check)
            backdrop_rgb = colour_rgb(backdrop or self.PALE)
            check_segments = (
                ((5.4, 11.2), (9.2, 14.8)),
                ((9.2, 14.8), (16.8, 7.2)))
            for y in range(height):
                for x in range(width):
                    totals = [0, 0, 0]
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
                                colour = check_rgb
                            elif inside(point_x, point_y, 1):
                                colour = fill_rgb
                            elif inside(point_x, point_y):
                                colour = outline_rgb
                            else:
                                colour = backdrop_rgb
                            for channel in range(3):
                                totals[channel] += colour[channel]
                    colour = tuple(
                        round(total / sample_count)
                        for total in totals)
                    image.put(
                        "#%02x%02x%02x" % colour,
                        (x, y))
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
        self._build_pipeline_tab()
        self._build_advanced_tab()
        self._build_help_tab()

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
        self._build_from_source_tab()

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
        self.input_text.bind(
            "<KeyRelease>",
            self._update_input_count)
        self.input_text.bind(
            "<FocusOut>",
            self._clear_text_selection,
            add="+")
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
        tab = self.from_source_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        self.source_options = ()
        self.source_options_by_label = {}
        self.source_options_by_key = {}
        self.source_display_labels_by_key = {}
        self.source_selected_label = tk.StringVar()
        self.source_language_label = tk.StringVar()
        self.source_chunk_size = tk.StringVar(value="30")
        self.source_concurrency = tk.StringVar(value="8")
        self.source_request_stagger_ms = tk.StringVar(value="100")
        self.source_allow_web_search = tk.BooleanVar(value=False)
        self.source_context_label = tk.StringVar(
            value=SOURCE_CONTEXT_LABELS["sentence"])
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
        self.source_preview_page_size = tk.StringVar(value="100")
        self.source_preview_status = tk.StringVar(
            value="Choose a prepared source, then load its local metadata.")
        self.source_preview_page_data = None
        self.source_preview_by_tree_id = {}

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
        self._build_source_jobs_page()
        self._load_source_catalogue()
        self._update_source_context_description()
        self._update_source_exclusion_controls()
        self._update_source_generate_button_state()

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
            lambda _event: self._schedule_source_estimate(),
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

        ttk.Label(
            config,
            text="WORDS PER OPENAI REQUEST",
            style="FieldLabel.TLabel").grid(
                row=5,
                column=0,
                sticky="w",
                padx=(0, 12))
        self.source_chunk_entry = ttk.Entry(
            config,
            textvariable=self.source_chunk_size,
            width=14,
            style="App.TEntry")
        self.source_chunk_entry.grid(
            row=5,
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
                row=5,
                column=2,
                sticky="w",
                padx=(12, 0))

        rate_controls = ttk.Frame(
            config,
            style="Panel.TFrame")
        rate_controls.grid(
            row=6,
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
        ttk.Label(
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
            wraplength=950).grid(
                row=7,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(7, 0))

        ttk.Checkbutton(
            config,
            text=(
                "Allow one web search per request when the meaning is unclear"
            ),
            variable=self.source_allow_web_search,
            command=self._schedule_source_estimate,
            style="Panel.TCheckbutton").grid(
                row=8,
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
                row=9,
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
        tk.Label(
            estimate,
            text="ESTIMATED OPENAI COST",
            background=self.PALE,
            foreground=self.TEXT_SECONDARY,
            font=("DejaVu Sans", 8, "bold")).grid(
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
        ttk.Checkbutton(
            actions,
            text=(
                "I authorise the paid OpenAI requests shown in this estimate"
            ),
            variable=self.source_paid_authorized,
            command=self._update_source_generate_button_state,
            style="Panel.TCheckbutton").grid(
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
                self.source_concurrency,
                self.source_request_stagger_ms):
            variable.trace_add(
                "write",
                self._schedule_source_estimate)
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
            value="Classical Chinese (Warring States)")
        self.source_file_use_gpu = tk.BooleanVar(value=True)
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
                "the CPU rather than failing when CUDA is unavailable."
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
            value="Classical Chinese (Warring States)")
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
        page = self.source_jobs_page
        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=3)
        page.rowconfigure(3, weight=2)
        self._source_header(
            page,
            "REQUEST JOBS AND FAILURES",
            (
                "Completed chunks are retained. Connection failures may be "
                "recovered automatically by the backend; invalid responses "
                "remain stopped until you inspect and explicitly retry them."
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
        columns = (
            "source",
            "chunk",
            "worker",
            "status",
            "attempts",
            "detail",
        )
        self.source_jobs_tree = ttk.Treeview(
            table,
            columns=columns,
            show="headings",
            selectmode="extended",
            style="Jobs.Treeview")
        headings = {
            "source": "Source",
            "chunk": "Chunk",
            "worker": "Worker",
            "status": "Status",
            "attempts": "Attempts",
            "detail": "Latest detail",
        }
        widths = {
            "source": 230,
            "chunk": 105,
            "worker": 90,
            "status": 125,
            "attempts": 75,
            "detail": 520,
        }
        for column in columns:
            self.source_jobs_tree.heading(
                column,
                text=headings[column])
            self.source_jobs_tree.column(
                column,
                width=widths[column],
                minwidth=60,
                stretch=column in {"source", "detail"})
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
        self.source_jobs_tree.tag_configure(
            "running",
            foreground=self.WARNING)
        self.source_jobs_tree.tag_configure(
            "completed",
            foreground=self.ACCENT)
        self.source_job_by_tree_id = {}

        actions = ttk.Frame(page, style="App.TFrame")
        actions.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(10, 0))
        actions.columnconfigure(4, weight=1)
        ttk.Button(
            actions,
            text="Refresh",
            command=self._refresh_source_jobs,
            style="Secondary.TButton",
            cursor="hand2").grid(
                row=0,
                column=0,
                sticky="w")
        self.source_inspect_button = ttk.Button(
            actions,
            text="Inspect selected",
            command=self._inspect_selected_source_job,
            style="Secondary.TButton",
            cursor="hand2")
        self.source_inspect_button.grid(
            row=0,
            column=1,
            sticky="w",
            padx=(8, 0))
        self.source_retry_selected_button = ttk.Button(
            actions,
            text="Retry selected…",
            command=self._retry_selected_source_jobs,
            state=(
                tk.NORMAL
                if self.source_retry_callback is not None
                else tk.DISABLED),
            style="Secondary.TButton",
            cursor="hand2")
        self.source_retry_selected_button.grid(
            row=0,
            column=2,
            sticky="w",
            padx=(8, 0))
        self.source_retry_all_button = ttk.Button(
            actions,
            text="Retry all failed…",
            command=self._retry_all_source_jobs,
            state=(
                tk.NORMAL
                if self.source_retry_callback is not None
                else tk.DISABLED),
            style="Secondary.TButton",
            cursor="hand2")
        self.source_retry_all_button.grid(
            row=0,
            column=3,
            sticky="w",
            padx=(8, 0))
        ttk.Label(
            actions,
            textvariable=self.source_action_status,
            style="Status.TLabel",
            wraplength=470).grid(
                row=0,
                column=4,
                sticky="e",
                padx=(12, 0))

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
        inspect_panel.rowconfigure(1, weight=1)
        ttk.Label(
            inspect_panel,
            text="INSPECTED REQUEST / RESPONSE",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                padx=7,
                pady=(4, 5))
        self.source_job_inspection = tk.Text(
            inspect_panel,
            height=8,
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
        inspect_scrollbar = RoundedScrollbar(
            inspect_panel,
            command=self.source_job_inspection.yview,
            background=self.PANEL_BACKGROUND,
            active=self.ACCENT)
        self.source_job_inspection.configure(
            yscrollcommand=inspect_scrollbar.set)
        self.source_job_inspection.grid(
            row=1,
            column=0,
            sticky="nsew")
        inspect_scrollbar.grid(
            row=1,
            column=1,
            sticky="ns")
        self.source_job_inspection.bind(
            "<FocusOut>",
            self._clear_text_selection,
            add="+")

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
            else None)
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
                try:
                    source_language = pipeline_store.get_language(
                        option.source_language_key)
                except ValueError:
                    self.source_language_label.set("")
                else:
                    self.source_language_label.set(source_language.name)
            summary = []
            if option.word_count is not None:
                summary.append(
                    f"{option.word_count:,} unique candidate words")
            if option.section_count is not None:
                summary.append(
                    f"{option.section_count:,} source sections")
            summary.append(
                "built-in preset"
                if option.preset
                else "locally prepared source")
            self.source_summary.set(" · ".join(summary))
            self.source_deck_notice.set(
                f'Creates and imports “Vocabulary from {option.name}”.')
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
        self._sync_learned_filter_controls()
        self._schedule_source_estimate()

    def _source_context_changed(self, _event=None):
        self._update_source_context_description()
        self.source_paid_authorized.set(False)
        self._schedule_source_estimate()

    def _update_source_context_description(self):
        key = SOURCE_CONTEXT_KEYS_BY_LABEL.get(
            self.source_context_label.get(),
            "sentence")
        self.source_context_description.set(
            SOURCE_CONTEXT_DESCRIPTIONS[key])

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

    def _manual_input_filter_request(self, text):
        candidates = tuple(
            line.strip()
            for line in text.splitlines()
            if line.strip())
        if not candidates:
            raise ValueError("Enter at least one nonblank line.")
        return {
            "text": text,
            "candidates": candidates,
            "anki_exclusions": self._selected_anki_exclusions(
                self.get_generation_language().key),
        }

    def _source_request(self):
        option = self._selected_source_option()
        chunk_size = parse_source_chunk_size(
            self.source_chunk_size.get())
        concurrency = parse_source_chunk_size(
            self.source_concurrency.get(),
            maximum=64)
        request_stagger_ms = parse_request_stagger_ms(
            self.source_request_stagger_ms.get())
        context_mode = SOURCE_CONTEXT_KEYS_BY_LABEL.get(
            self.source_context_label.get())
        if context_mode is None:
            raise ValueError("Select a source context option.")
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
        if selected_language_key:
            source_pipelines = []
            for pipeline in pipelines:
                settings = pipeline_store.get_language_settings(
                    pipeline,
                    selected_language_key)
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
            "context_mode": context_mode,
            "concurrency": concurrency,
            "request_stagger_ms": request_stagger_ms,
            "max_transient_retries": 3,
            "allow_web_search": bool(
                getattr(
                    self,
                    "source_allow_web_search",
                    False).get()
                if hasattr(
                    getattr(self, "source_allow_web_search", None),
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
            self.source_estimate_price.set(price)
            self.source_estimate_detail.set(detail)
            if (
                    self._estimate_candidate_count(value) == 0
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
    def _estimate_limit_warning(estimate):
        if estimate is None:
            return None
        if isinstance(estimate, dict):
            return estimate.get("request_limit_warning")
        return getattr(estimate, "request_limit_warning", None)

    def _update_source_generate_button_state(self):
        if not hasattr(self, "source_generate_button"):
            return
        candidate_count = self._estimate_candidate_count(
            self.source_estimate_result)
        available = (
            self.source_generate_callback is not None
            and self.source_estimate_result is not None
            and candidate_count != 0
            and not self._estimate_limit_warning(
                self.source_estimate_result)
            and self.source_paid_authorized.get()
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
        if not self.source_paid_authorized.get():
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
        candidate_count = self._estimate_candidate_count(
            self.source_estimate_result)
        if candidate_count == 0:
            messagebox.showinfo(
                "No new vocabulary",
                (
                    "Every candidate word is already present in the selected "
                    "Anki field. No OpenAI request or deck was created."
                ),
                parent=self.root)
            return
        request["paid_confirmed"] = True
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
            "inspect": "Loading saved request and response details…",
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

        if action == "inspect":
            self._show_source_job_inspection(value)
            self.source_action_status.set(
                "Loaded the saved request and response.")
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
        for item_id in self.source_jobs_tree.get_children():
            self.source_jobs_tree.delete(item_id)
        self.source_job_by_tree_id = {}
        for index, job in enumerate(jobs):
            def read(name, default=""):
                if isinstance(job, dict):
                    return job.get(name, default)
                return getattr(job, name, default)

            job_id = str(read("job_id", read("id", index)))
            source = str(read(
                "source_name",
                read("source", "")))
            chunk = read(
                "chunk_label",
                read("chunk", read("chunk_index", "")))
            worker = read(
                "worker",
                read("worker_id", "—"))
            status = str(read("status", "unknown"))
            attempts = read("attempts", read("attempt_count", 0))
            detail = str(read(
                "detail",
                read("error", read("message", ""))))
            tree_id = f"source_job_{index}"
            self.source_jobs_tree.insert(
                "",
                tk.END,
                iid=tree_id,
                values=(
                    source,
                    chunk,
                    worker,
                    status,
                    attempts,
                    detail,
                ),
                tags=(status.lower(),))
            self.source_job_by_tree_id[tree_id] = {
                "job_id": job_id,
                "status": status,
                "record": job,
            }
        self.source_action_status.set(
            f"{len(jobs):,} saved source jobs loaded.")

    def _selected_source_jobs(self):
        return tuple(
            self.source_job_by_tree_id[item_id]
            for item_id in self.source_jobs_tree.selection()
            if item_id in self.source_job_by_tree_id)

    def _inspect_selected_source_job(self):
        selected = self._selected_source_jobs()
        if len(selected) != 1:
            messagebox.showinfo(
                "Select one job",
                "Select exactly one request to inspect.",
                parent=self.root)
            return
        record = selected[0]
        if self.source_inspect_callback is None:
            self._show_source_job_inspection(record["record"])
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
        self._show_source_job_inspection(value)
        self.source_action_status.set(
            "Loaded the saved request and response.")

    def _show_source_job_inspection(self, value):
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                default=str)
        else:
            text = str(value)
        self.source_job_inspection.configure(state=tk.NORMAL)
        self.source_job_inspection.delete("1.0", tk.END)
        self.source_job_inspection.insert("1.0", text)
        self.source_job_inspection.configure(state=tk.DISABLED)

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
        if not messagebox.askyesno(
                "Authorize paid retries?",
                (
                    f"Resume or retry {len(records):,} request(s)? These "
                    "requests may incur additional OpenAI charges. Completed "
                    "requests will not be repeated."
                ),
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
        if (
                hasattr(self, "generate_notebook")
                and self.generate_notebook.select()
                == str(self.from_source_tab)):
            self._schedule_source_estimate()

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
            self._refresh_source_jobs(show_errors=False)

    def _build_pipeline_tab(self):
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

    def _build_advanced_tab(self):
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
            self.prompt_options[0].key
            if self.prompt_options
            else "")
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

    def _build_help_tab(self):
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
        pipelines = tuple(
            row.to_config()
            for row in self.pipeline_rows)
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
            self._refresh_prompt_selector_display()
            self.root.after_idle(
                self._refresh_prompt_selector_display)
            return
        if selected_tab == str(self.generate_tab):
            if (
                    hasattr(self, "generate_notebook")
                    and self.generate_notebook.select()
                    == str(self.from_source_tab)):
                self._schedule_source_estimate()
            return
        if selected_tab != str(self.pipeline_tab):
            return
        self.deck_status.set(
            "Connecting to Anki and refreshing decks…")
        self.refresh_anki_decks(launch_if_needed=True)

    def get_generation_language(self):
        selected_name = self.generation_language.get()
        for language in pipeline_store.list_languages():
            if language.name == selected_name:
                return language
        raise ValueError("Select an input language on the Generate tab.")

    def _generation_language_changed(self, _event=None):
        language = self.get_generation_language()
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
            elif not client.is_available():
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
        if self.save_pipeline_rows(show_errors=True) is None:
            return
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

        if self._language_filter(
                self.get_generation_language().key).enabled:
            if self.manual_input_filter_callback is None:
                messagebox.showerror(
                    "Learned-word filter unavailable",
                    (
                        "This build has not connected the manual-input "
                        "learned-word filter. Disable it or reconnect the "
                        "filter backend."
                    ),
                    parent=self.root)
                return
            try:
                filter_request = self._manual_input_filter_request(words)
            except ValueError as error:
                messagebox.showerror(
                    "Learned-word filter is incomplete",
                    str(error),
                    parent=self.root)
                return
            self._set_generation_busy(True)
            self._set_status(
                "Checking exact input lines against Anki before generation…",
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
            summary = self.pipeline_executor(
                words,
                pipelines,
                progress_callback=lambda progress: (
                    self.result_queue.put(
                        ("pipeline_progress", progress))))
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
                    f"Omitted {result.excluded_count:,} learned lines; "
                    f"starting {len(pipelines)} generation pipelines for "
                    f"{result.remaining_count:,} remaining lines…"
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
