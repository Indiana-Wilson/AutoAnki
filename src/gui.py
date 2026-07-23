import queue
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk

import anki_integration
import credential_store
import pipeline_runner
import pipeline_store
import process_text


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
                self._wheel_scroll,
                add="+")
            widget.bind(
                "<Button-4>",
                self._wheel_scroll,
                add="+")
            widget.bind(
                "<Button-5>",
                self._wheel_scroll,
                add="+")
            stack.extend(widget.winfo_children())


class PipelineEditor:
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

        for language in pipeline_store.list_languages():
            self._add_language_tab(language, config)

        active_tab = self.language_tabs.get(config.language_key)
        if active_tab is not None:
            self.language_notebook.select(active_tab)
        self.last_active_language_key = config.language_key
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
            separate = self.separate_target_deck_variables[
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
            value=settings.separate_target_decks)
        self.separate_target_deck_variables[
            language.key] = separate_target_decks
        split_decks_button = ttk.Checkbutton(
            destination,
            text="Send each card type to a separate Anki deck",
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
        active_language = self.get_active_language()
        language_settings = []
        for language in pipeline_store.list_languages():
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
                        self.separate_target_deck_variables[
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
            pipeline_saver=None):
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
        self.result_queue = queue.Queue()
        self.deck_result_queue = queue.Queue()
        self.pipeline_rows = []
        self.pipeline_save_after_id = None
        self.pipeline_notice_after_id = None
        self.deck_refresh_in_progress = False
        self.deck_launch_requested = False
        self.generation_in_progress = False

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
            loaded_pipelines = tuple(self.pipeline_loader())
            self.pipeline_configs = (
                loaded_pipelines[:1]
                or (pipeline_store.default_pipeline(),))
            self.pipeline_load_error = None
        except (OSError, ValueError) as error:
            self.pipeline_configs = (
                pipeline_store.default_pipeline(),)
            self.pipeline_load_error = error

        configured_decks = {
            pipeline.target_deck
            for pipeline in self.pipeline_configs
            if pipeline.target_deck
        }
        configured_decks.update(
            target_deck
            for pipeline in self.pipeline_configs
            for _card_type_key, target_deck
            in pipeline.card_type_target_decks
            if target_deck)
        configured_decks.update(
            settings.target_deck
            for pipeline in self.pipeline_configs
            for settings in pipeline.language_settings
            if settings.target_deck)
        configured_decks.update(
            target_deck
            for pipeline in self.pipeline_configs
            for settings in pipeline.language_settings
            for _card_type_key, target_deck
            in settings.card_type_target_decks
            if target_deck)
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
                check="#FFFFFF"):
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
            backdrop_rgb = colour_rgb(self.PALE)
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
        self.notebook.add(
            self.generate_tab,
            text="Generate")
        self.notebook.add(
            self.pipeline_tab,
            text="Card setup")
        self.notebook.add(
            self.advanced_tab,
            text="Advanced")

        self._build_generate_tab()
        self._build_pipeline_tab()
        self._build_advanced_tab()

    def _build_generate_tab(self):
        tab = self.generate_tab
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
            textvariable=self.input_count,
            style="Muted.TLabel").grid(
                row=0,
                column=1,
                sticky="e")

        ttk.Label(
            editor,
            text=(
                "The same input is sent through every configured pipeline. "
                "Each pipeline makes one OpenAI API request."),
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

    def _build_pipeline_tab(self):
        tab = self.pipeline_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        header_surface = RoundedPanel(
            tab,
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
            tab,
            style="App.TFrame")
        self.pipeline_editor_container.grid(
            row=1,
            column=0,
            sticky="nsew",
            pady=(10, 0))
        self.pipeline_editor_container.columnconfigure(0, weight=1)
        self.pipeline_editor_container.rowconfigure(0, weight=1)

        ttk.Label(
            tab,
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
            text="PROMPT EDITOR",
            background=self.PALE,
            foreground=self.TEXT_PRIMARY,
            font=("DejaVu Sans", 11, "bold")).grid(
                row=0,
                column=0,
                sticky="w")
        tk.Label(
            header,
            text=(
                "Inspect the instructions sent to OpenAI. Changes affect "
                "future paid requests, so editing is locked by default."),
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

        self.prompt_options = pipeline_store.discover_prompts(
            process_text.PROJECT_ROOT)
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
            text="PROMPT",
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
            text="Save prompt",
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
                "No prompt files were found in input/prompts.")
            self.prompt_selector.configure(state=tk.DISABLED)
            self.prompt_editing_checkbox.configure(state=tk.DISABLED)

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
                text,
                process_text.PROJECT_ROOT)
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
        self._update_pipeline_status()

    def _update_pipeline_status(self, message=None):
        if message is None:
            try:
                config = self.pipeline_rows[0].to_config()
                language = pipeline_store.get_language(
                    config.language_key)
                output_count = len(config.card_type_keys)
                if output_count:
                    output_word = (
                        "output" if output_count == 1 else "outputs")
                    deck_mode = (
                        "separate target decks"
                        if config.separate_target_decks
                        else "one target deck")
                    message = (
                        f"{language.name} · {output_count} card "
                        f"{output_word} selected · {deck_mode} · "
                        "1 paid API request per run")
                else:
                    message = (
                        f"{language.name} · Select at least one card output.")
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

        available_prompt_keys = {
            prompt.key
            for prompt in pipeline_store.discover_prompts(
                process_text.PROJECT_ROOT)
        }
        for pipeline in pipelines:
            prompt_key = pipeline_store.get_prompt_key(pipeline)
            if prompt_key not in available_prompt_keys:
                raise ValueError(
                    f'Prompt "{prompt_key}" was not found.')
        return pipelines

    def schedule_pipeline_save(self, *_args):
        if self.pipeline_save_after_id is not None:
            self.root.after_cancel(self.pipeline_save_after_id)
        self.pipeline_save_after_id = self.root.after(
            400,
            self._autosave_pipeline_rows)

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
        if selected_tab != str(self.pipeline_tab):
            return
        self.deck_status.set(
            "Connecting to Anki and refreshing decks…")
        self.refresh_anki_decks(launch_if_needed=True)

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

        if not self._ensure_api_key():
            return

        self._set_generation_busy(True)
        self._set_status(
            f"Starting {len(pipelines)} generation pipelines…",
            self.WARNING)

        worker = threading.Thread(
            target=self._generate_in_background,
            args=(words, pipelines),
            daemon=True)
        worker.start()
        self.root.after(
            self.POLL_INTERVAL_MS,
            self._poll_result)

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
                    f"{process_text.PROJECT_ROOT / 'output'}")

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
    root = tk.Tk()
    AutoAnkiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
