import queue
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import simpledialog
from tkinter import ttk

import anki_integration
import credential_store
import pipeline_runner
import pipeline_store
import process_text


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
    }

    def __init__(self, app, parent, config):
        self.app = app
        self.config = config
        self.frame = ttk.Frame(
            parent,
            padding=(18, 16),
            style="Panel.TFrame")
        self.frame.grid(row=0, column=0, sticky="nsew")
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)

        self.target_deck = tk.StringVar(
            value=config.target_deck)
        self.language_tabs = {}
        self.language_by_tab = {}
        self.card_output_variables = {}
        self.card_output_traces = []

        ttk.Label(
            self.frame,
            text="CHOOSE A LANGUAGE",
            style="Section.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                pady=(0, 10))

        self.language_notebook = ttk.Notebook(
            self.frame,
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

        destination = ttk.Frame(
            self.frame,
            padding=(0, 18, 0, 0),
            style="Panel.TFrame")
        destination.grid(row=2, column=0, sticky="ew")
        destination.columnconfigure(0, weight=1)
        ttk.Label(
            destination,
            text="TARGET ANKI DECK",
            style="FieldLabel.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
                pady=(0, 6))

        self.target_deck_box = ttk.Combobox(
            destination,
            textvariable=self.target_deck,
            width=80,
            values=app.deck_options)
        self.target_deck_box.grid(
            row=1,
            column=0,
            sticky="ew")

        self.variable_traces = (
            (self.target_deck, self.target_deck.trace_add(
                "write", app.schedule_pipeline_save)),
        )

    def _add_language_tab(self, language, config):
        tab = ttk.Frame(
            self.language_notebook,
            padding=(20, 18),
            style="Panel.TFrame")
        tab.columnconfigure(0, weight=1)
        self.language_notebook.add(tab, text=language.name)
        self.language_tabs[language.key] = tab
        self.language_by_tab[str(tab)] = language

        ttk.Label(
            tab,
            text="Cards to create",
            style="Section.TLabel").grid(
                row=0,
                column=0,
                sticky="w")
        ttk.Label(
            tab,
            text=(
                "Select one or more outputs. They will share a single "
                "OpenAI response."),
            style="Muted.TLabel").grid(
                row=1,
                column=0,
                sticky="w",
                pady=(3, 14))

        selected_keys = (
            set(config.card_type_keys)
            if config.language_key == language.key
            else {language.detailed_card_type_key})
        variables = {}
        for row_number, (card_type_key, label) in enumerate(
                language.card_types,
                start=2):
            option = ttk.Frame(
                tab,
                padding=(12, 9),
                style="App.TFrame")
            option.grid(
                row=row_number,
                column=0,
                sticky="ew",
                pady=(0, 7))
            option.columnconfigure(0, weight=1)

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
            ttk.Label(
                option,
                text=self.CARD_DESCRIPTIONS[label],
                style="CardDescription.TLabel",
                wraplength=980).grid(
                    row=1,
                    column=0,
                    sticky="w",
                    padx=(24, 0),
                    pady=(2, 0))
            trace_id = variable.trace_add(
                "write",
                self._card_selection_changed)
            self.card_output_traces.append((variable, trace_id))

        self.card_output_variables[language.key] = variables

    def _language_changed(self, _event=None):
        language = self.get_active_language()
        if language.key == self.last_active_language_key:
            return
        self.last_active_language_key = language.key
        self.app.schedule_pipeline_save()
        self.app._update_pipeline_status()

    def _card_selection_changed(self, *_args):
        self.app.schedule_pipeline_save()
        self.app._update_pipeline_status()

    def get_active_language(self):
        selected_tab = self.language_notebook.select()
        language = self.language_by_tab.get(selected_tab)
        if language is None:
            raise ValueError("Select a language.")
        return language

    def to_config(self):
        language = self.get_active_language()
        variables = self.card_output_variables[language.key]
        card_type_keys = tuple(
            card_type_key
            for card_type_key, _label in language.card_types
            if variables[card_type_key].get())

        return replace(
            self.config,
            language_key=language.key,
            card_type_keys=card_type_keys,
            target_deck=self.target_deck.get().strip())

    def update_deck_options(self, deck_options):
        self.target_deck_box.configure(values=deck_options)

    def destroy(self):
        for variable, trace_id in self.card_output_traces:
            variable.trace_remove("write", trace_id)
        for variable, trace_id in self.variable_traces:
            variable.trace_remove("write", trace_id)
        self.frame.destroy()


class AutoAnkiApp:
    POLL_INTERVAL_MS = 100
    DECK_REFRESH_INTERVAL_MS = 5000
    WINDOW_BACKGROUND = "#f3f5f9"
    PANEL_BACKGROUND = "#ffffff"
    TEXT_PRIMARY = "#172033"
    TEXT_SECONDARY = "#667085"
    ACCENT = "#4267c7"

    def __init__(
            self,
            root,
            pipeline_executor=None,
            pipeline_loader=None,
            pipeline_saver=None):
        self.root = root
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
        self.deck_refresh_in_progress = False

        root.title("AutoAnki")
        root.configure(background=self.WINDOW_BACKGROUND)

        self.status = tk.StringVar(value="Ready to generate")
        self.output = tk.StringVar(
            value=str(process_text.DECK_PATH))
        self.input_count = tk.StringVar(
            value="0 lines · 0 characters")
        self.pipeline_status = tk.StringVar()

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

        self.deck_options = tuple(sorted({
            pipeline.target_deck
            for pipeline in self.pipeline_configs
        }))

        self._configure_styles()
        self._build_widgets()
        self._render_pipeline_rows()
        self._set_initial_window_size()
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
        self.root.minsize(min(780, width), min(600, height))

    def _configure_styles(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure(
            "App.TFrame",
            background=self.WINDOW_BACKGROUND)
        style.configure(
            "Panel.TFrame",
            background=self.PANEL_BACKGROUND)
        style.configure(
            "Title.TLabel",
            background=self.WINDOW_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            font=("TkDefaultFont", 22, "bold"))
        style.configure(
            "Subtitle.TLabel",
            background=self.WINDOW_BACKGROUND,
            foreground=self.TEXT_SECONDARY,
            font=("TkDefaultFont", 10))
        style.configure(
            "Section.TLabel",
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            font=("TkDefaultFont", 11, "bold"))
        style.configure(
            "Muted.TLabel",
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_SECONDARY,
            font=("TkDefaultFont", 9))
        style.configure(
            "Status.TLabel",
            background=self.WINDOW_BACKGROUND,
            foreground=self.TEXT_SECONDARY,
            font=("TkDefaultFont", 10))
        style.configure(
            "FieldLabel.TLabel",
            background=self.PANEL_BACKGROUND,
            foreground=self.TEXT_SECONDARY,
            font=("TkDefaultFont", 10))
        style.configure(
            "CardDescription.TLabel",
            background=self.WINDOW_BACKGROUND,
            foreground=self.TEXT_SECONDARY,
            font=("TkDefaultFont", 9))
        style.configure(
            "CardOption.TCheckbutton",
            background=self.WINDOW_BACKGROUND,
            foreground=self.TEXT_PRIMARY,
            font=("TkDefaultFont", 10, "bold"))
        style.map(
            "CardOption.TCheckbutton",
            background=[
                ("active", self.WINDOW_BACKGROUND),
                ("selected", self.WINDOW_BACKGROUND)],
            foreground=[
                ("disabled", self.TEXT_SECONDARY),
                ("active", self.TEXT_PRIMARY)])
        style.configure(
            "Accent.TButton",
            background=self.ACCENT,
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            padding=(18, 10),
            font=("TkDefaultFont", 10, "bold"))
        style.map(
            "Accent.TButton",
            background=[
                ("disabled", "#aab5cf"),
                ("pressed", "#3153a8"),
                ("active", "#5075d3")],
            foreground=[("disabled", "#eef1f7")])
        style.configure(
            "Secondary.TButton",
            background="#ffffff",
            foreground=self.TEXT_PRIMARY,
            bordercolor="#d2d8e3",
            padding=(12, 8),
            font=("TkDefaultFont", 9))
        style.map(
            "Secondary.TButton",
            background=[
                ("pressed", "#e9edf5"),
                ("active", "#f6f8fb")])
        style.configure(
            "TNotebook",
            background=self.WINDOW_BACKGROUND,
            borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            padding=(16, 8),
            font=("TkDefaultFont", 10, "bold"))
        style.map(
            "TNotebook.Tab",
            padding=[
                ("selected", (20, 13)),
                ("!selected", (16, 8))],
            background=[
                ("selected", self.PANEL_BACKGROUND),
                ("!selected", "#d8d5ce")],
            foreground=[
                ("selected", self.TEXT_PRIMARY),
                ("!selected", self.TEXT_SECONDARY)])

    def _build_widgets(self):
        frame = ttk.Frame(
            self.root,
            padding=(28, 24, 28, 22),
            style="App.TFrame")
        frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        title = ttk.Label(
            frame,
            text="AutoAnki",
            style="Title.TLabel")
        title.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            frame,
            text=(
                "Turn notes into cards using the language and formats "
                "you choose."),
            style="Subtitle.TLabel")
        subtitle.grid(row=1, column=0, sticky="w", pady=(3, 14))

        self.notebook = ttk.Notebook(frame)
        self.notebook.grid(row=2, column=0, sticky="nsew")

        self.generate_tab = ttk.Frame(
            self.notebook,
            padding=(4, 16, 4, 4),
            style="App.TFrame")
        self.pipeline_tab = ttk.Frame(
            self.notebook,
            padding=(4, 16, 4, 4),
            style="App.TFrame")
        self.notebook.add(
            self.generate_tab,
            text="Generate")
        self.notebook.add(
            self.pipeline_tab,
            text="Card setup")

        self._build_generate_tab()
        self._build_pipeline_tab()

    def _build_generate_tab(self):
        tab = self.generate_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        editor = ttk.Frame(
            tab,
            padding=(18, 16),
            style="Panel.TFrame")
        editor.grid(row=0, column=0, sticky="nsew")
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

        self.input_text = scrolledtext.ScrolledText(
            editor,
            wrap=tk.WORD,
            undo=True,
            font=("TkDefaultFont", 11),
            background="#fbfcfe",
            foreground=self.TEXT_PRIMARY,
            insertbackground=self.TEXT_PRIMARY,
            selectbackground="#cbd8f5",
            selectforeground=self.TEXT_PRIMARY,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#d9dee8",
            highlightcolor=self.ACCENT,
            padx=12,
            pady=10)
        self.input_text.grid(row=2, column=0, sticky="nsew")
        self.input_text.bind(
            "<KeyRelease>",
            self._update_input_count)
        self.input_text.focus_set()

        controls = ttk.Frame(
            tab,
            style="App.TFrame")
        controls.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(16, 0))
        controls.columnconfigure(1, weight=1)

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

        ttk.Label(
            controls,
            textvariable=self.status,
            wraplength=540,
            style="Status.TLabel").grid(
                row=0,
                column=1,
                sticky="w",
                padx=(14, 0))

        self.api_key_button = ttk.Menubutton(
            controls,
            text="API key",
            style="Secondary.TButton",
            cursor="hand2")
        self.api_key_button.grid(
            row=0,
            column=2,
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

        output_panel = ttk.Frame(
            tab,
            padding=(14, 10),
            style="Panel.TFrame")
        output_panel.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(16, 0))
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

        header = ttk.Frame(
            tab,
            padding=(18, 14),
            style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="CARD GENERATION",
            style="Section.TLabel").grid(
                row=0,
                column=0,
                sticky="w")
        ttk.Label(
            header,
            text=(
                "Choose the language of your input, then select every card "
                "format you want to create."),
            style="Muted.TLabel",
            wraplength=880).grid(
                row=1,
                column=0,
                sticky="w",
                pady=(4, 0))

        self.pipeline_editor_container = ttk.Frame(
            tab,
            style="Panel.TFrame")
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
                    message = (
                        f"{language.name} · {output_count} card "
                        f"{output_word} selected · "
                        "1 paid API request per run")
                else:
                    message = (
                        f"{language.name} · Select at least one card output.")
            except (IndexError, ValueError):
                message = (
                    "Select a language and at least one card output.")
        self.pipeline_status.set(message)

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
            return None

        self.pipeline_configs = pipelines
        for row, pipeline in zip(
                self.pipeline_rows,
                pipelines):
            row.config = pipeline
        self._update_pipeline_status("Card settings saved automatically.")
        if show_confirmation:
            messagebox.showinfo(
                "Pipelines saved",
                "Your pipeline selections have been saved.",
                parent=self.root)
        return pipelines

    def refresh_anki_decks(self):
        if self.deck_refresh_in_progress:
            return
        self.deck_refresh_in_progress = True
        worker = threading.Thread(
            target=self._load_anki_decks_in_background,
            daemon=True)
        worker.start()
        self.root.after(
            self.POLL_INTERVAL_MS,
            self._poll_deck_result)

    def _load_anki_decks_in_background(self):
        try:
            client = anki_integration.AnkiConnectClient(timeout=3)
            if not client.is_available():
                self.deck_result_queue.put(("unavailable", None))
                return
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
            if value != self.deck_options:
                self.deck_options = value
                for row in self.pipeline_rows:
                    row.update_deck_options(value)
                self._update_pipeline_status()

        self.root.after(
            self.DECK_REFRESH_INTERVAL_MS,
            self.refresh_anki_decks)

    def _update_input_count(self, _event=None):
        value = self.input_text.get("1.0", "end-1c")
        line_count = len(value.splitlines()) if value else 0
        self.input_count.set(
            f"{line_count} lines · {len(value)} characters")

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

        self.generate_button.configure(state=tk.DISABLED)
        self.status.set(
            f"Starting {len(pipelines)} generation pipelines…")

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

        self.status.set("API key saved on this device.")
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
            self.status.set(
                "Saved API key removed. OPENAI_API_KEY is still active.")
        else:
            self.status.set("Saved API key removed.")

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
            self.status.set(
                f"Pipeline {value.index}/{value.total}: {action}…")
            self.root.after(
                self.POLL_INTERVAL_MS,
                self._poll_result)
            return

        self.generate_button.configure(state=tk.NORMAL)
        if outcome == "run_error":
            self.status.set("Pipeline run failed.")
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
            self.status.set(
                f"{success_count} pipelines succeeded; "
                f"{failure_count} failed.")
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

        self.status.set(
            f"Imported {summary.cards_moved} cards through "
            f"{success_count} pipelines.")
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
