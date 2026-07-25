"""High-level corpus orchestration suitable for both CLI and future GUI use."""

from dataclasses import dataclass
import json
from pathlib import Path

from corpus_pipeline.audit import audit_snapshot
from corpus_pipeline.catalogue import get_corpus_spec
from corpus_pipeline.cleaners import (
    CLEANER_VERSION,
    clean_corpus_pages,
)
from corpus_pipeline.contexts import (
    assemble_sections,
    with_section_titles,
)
from corpus_pipeline.mediawiki import fetch_corpus_pages
from corpus_pipeline.models import (
    CORPUS_SCHEMA_VERSION,
    BuildConfig,
    CorpusBuild,
    CorpusSnapshot,
)
from corpus_pipeline.processing import build_vocabulary
from corpus_pipeline.refiners import (
    JiebaLongSpanRefiner,
    refine_build_long_spans,
)
from corpus_pipeline.resources import (
    ensure_jieba_traditional_dictionary,
)
from corpus_pipeline.storage import (
    read_build,
    read_snapshot,
    resolve_corpus_base,
    snapshot_id,
    write_build,
    write_snapshot,
)
from corpus_pipeline.tokenizers import (
    CkipHanTokenizer,
    recommended_hardware_batch_size,
)
import runtime_paths


_AUTOMATIC_REFINER = object()


@dataclass(frozen=True)
class ProgressEvent:
    phase: str
    current: int
    total: int
    message: str


@dataclass(frozen=True)
class SnapshotResult:
    snapshot: CorpusSnapshot
    path: Path
    from_cache: bool


@dataclass(frozen=True)
class BuildResult:
    build: CorpusBuild
    path: Path


def _expected_section_ids(spec):
    return tuple(
        f"{spec.key}:chapter:{number:03d}"
        for number in range(1, spec.expected_section_count + 1)
    )


def _is_local_custom_snapshot(snapshot):
    """Identify snapshots produced by the registered local-source workflow."""
    return (
        snapshot.spec_key.startswith("custom-")
        and snapshot.edition.startswith("Local document:")
        and bool(snapshot.pages)
        and all(page.url.startswith("local:") for page in snapshot.pages)
    )


def make_snapshot(spec, pages):
    """Clean fetched pages and assign exact global offsets."""
    source_pages = tuple(pages)
    raw_sections = clean_corpus_pages(spec, source_pages)
    sections, canonical_text = assemble_sections(raw_sections)
    snapshot = CorpusSnapshot(
        spec_key=spec.key,
        edition=spec.edition,
        source_language_key=spec.source_language_key,
        pages=source_pages,
        sections=sections,
        canonical_text=canonical_text,
        cleaner_version=CLEANER_VERSION)
    audit_snapshot(
        snapshot,
        expected_section_count=spec.expected_section_count,
        expected_section_ids=_expected_section_ids(spec),
        require_no_latin=True)
    return snapshot


def default_tokenizer(spec):
    options = {
        "device": "auto",
        "batch_size": recommended_hardware_batch_size(),
    }
    if spec.source_language_key in {
            "classical_chinese_han",
            "classical_chinese_wang_bi",
            "classical_chinese_warring_states"}:
        return CkipHanTokenizer.for_shanggu(**options)
    if spec.source_language_key == "classical_chinese_ming":
        return CkipHanTokenizer.for_jindai(**options)
    raise KeyError(
        f"No default corpus tokenizer for {spec.source_language_key!r}.")


def default_refiner(spec, corpus_root):
    """Return the pinned long-span fallback needed by one built-in work."""
    if spec.long_span_refiner is None:
        return None
    if spec.long_span_refiner != "jieba_traditional_max4":
        raise KeyError(
            f"Unknown long-span refinement policy for {spec.key!r}.")
    dictionary_path = ensure_jieba_traditional_dictionary(corpus_root)
    return JiebaLongSpanRefiner(dictionary_path)


class CorpusService:
    """Prepare reproducible vocabulary runs without OpenAI or Anki."""

    def __init__(self, corpus_root=None, progress_callback=None):
        self.corpus_root = Path(
            corpus_root
            or runtime_paths.get_corpus_output_directory())
        self.progress_callback = progress_callback

    def _emit(self, phase, current, total, message):
        if self.progress_callback is not None:
            self.progress_callback(ProgressEvent(
                phase=phase,
                current=current,
                total=total,
                message=message))

    def _latest_snapshot_path(self, spec):
        base = resolve_corpus_base(
            self.corpus_root,
            spec.key)
        pointer_path = base / "latest_snapshot.json"
        if not pointer_path.is_file():
            return None
        pointer = json.loads(
            pointer_path.read_text(encoding="utf-8"))
        if pointer.get("schema_version") != CORPUS_SCHEMA_VERSION:
            raise ValueError(
                "Latest snapshot pointer uses an unsupported schema.")
        expected_identifier = pointer.get("snapshot_id")
        if not isinstance(expected_identifier, str):
            raise ValueError(
                "Latest snapshot pointer has no valid snapshot ID.")
        relative_path = pointer.get("relative_path")
        if not isinstance(relative_path, str):
            raise ValueError("Latest snapshot pointer is invalid.")
        candidate = (base / relative_path).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as error:
            raise ValueError(
                "Latest snapshot pointer escapes corpus storage.") from error
        if not candidate.is_dir():
            raise ValueError("Latest corpus snapshot is missing.")
        if candidate.name != expected_identifier:
            raise ValueError(
                "Latest snapshot path and identity do not agree.")
        return candidate

    def load_latest_snapshot(self, spec_key):
        spec = get_corpus_spec(spec_key)
        path = self._latest_snapshot_path(spec)
        if path is None:
            raise FileNotFoundError(
                f"No saved snapshot exists for {spec.title}.")
        snapshot = read_snapshot(path)
        if snapshot.spec_key != spec.key:
            raise ValueError("Saved snapshot belongs to another corpus.")
        pointer = json.loads(
            (
                resolve_corpus_base(
                    self.corpus_root,
                    spec.key)
                / "latest_snapshot.json"
            ).read_text(encoding="utf-8"))
        if snapshot_id(snapshot) != pointer["snapshot_id"]:
            raise ValueError(
                "Latest snapshot content and identity do not agree.")
        audit_snapshot(
            snapshot,
            expected_section_count=spec.expected_section_count,
            expected_section_ids=_expected_section_ids(spec),
            require_no_latin=True)
        if snapshot.cleaner_version != CLEANER_VERSION:
            raise ValueError(
                "The saved snapshot uses an older cleaner. Run the fetch "
                "command once to rebuild it locally from the saved raw "
                "wikitext.")
        return SnapshotResult(snapshot, path, True)

    def fetch(
            self,
            spec_key,
            *,
            refresh=False,
            client=None,
            **fetch_options):
        """Fetch once, or reuse the last validated raw snapshot by default."""
        spec = get_corpus_spec(spec_key)
        if not refresh:
            cached_path = self._latest_snapshot_path(spec)
            if cached_path is not None:
                cached_snapshot = read_snapshot(cached_path)
                if cached_snapshot.spec_key != spec.key:
                    raise ValueError(
                        "Saved snapshot belongs to another corpus.")
                if snapshot_id(cached_snapshot) != cached_path.name:
                    raise ValueError(
                        "Latest snapshot path and content identity do not "
                        "agree.")
                audit_snapshot(
                    cached_snapshot,
                    expected_section_count=spec.expected_section_count,
                    expected_section_ids=_expected_section_ids(spec),
                    require_no_latin=True)
                if cached_snapshot.cleaner_version == CLEANER_VERSION:
                    self._emit(
                        "fetch",
                        1,
                        1,
                        f"Using saved {spec.title} source snapshot.")
                    return SnapshotResult(
                        cached_snapshot,
                        cached_path,
                        True)

                self._emit(
                    "clean",
                    0,
                    spec.expected_section_count,
                    f"Re-cleaning saved {spec.title} raw revisions.")
                snapshot = make_snapshot(
                    spec,
                    cached_snapshot.pages)
                path = write_snapshot(
                    snapshot,
                    self.corpus_root,
                    expected_section_count=spec.expected_section_count,
                    expected_section_ids=_expected_section_ids(spec),
                    require_no_latin=True)
                snapshot = read_snapshot(path)
                self._emit(
                    "clean",
                    spec.expected_section_count,
                    spec.expected_section_count,
                    f"Saved updated {spec.title} snapshot.")
                return SnapshotResult(snapshot, path, True)

        self._emit(
            "fetch",
            0,
            len(spec.requested_titles),
            f"Fetching {spec.title} from Chinese Wikisource.")

        def fetch_progress(current, total):
            self._emit(
                "fetch",
                current,
                total,
                f"Fetched {current} of {total} source pages.")

        pages = fetch_corpus_pages(
            spec,
            client=client,
            progress_callback=fetch_progress,
            **fetch_options)
        self._emit(
            "clean",
            0,
            spec.expected_section_count,
            f"Cleaning and validating {spec.title}.")
        snapshot = make_snapshot(spec, pages)
        path = write_snapshot(
            snapshot,
            self.corpus_root,
            expected_section_count=spec.expected_section_count,
            expected_section_ids=_expected_section_ids(spec),
            require_no_latin=True)
        snapshot = read_snapshot(path)
        self._emit(
            "clean",
            spec.expected_section_count,
            spec.expected_section_count,
            f"Saved validated {spec.title} snapshot.")
        return SnapshotResult(snapshot, path, False)

    def build(
            self,
            spec_key,
            *,
            snapshot=None,
            tokenizer=None,
            refiner=_AUTOMATIC_REFINER,
            config=None):
        """Tokenize a saved snapshot and publish an audited local run."""
        spec = get_corpus_spec(spec_key)
        if snapshot is None:
            snapshot = self.load_latest_snapshot(spec_key).snapshot
        elif isinstance(snapshot, (str, Path)):
            snapshot = read_snapshot(snapshot)
        if snapshot.spec_key != spec.key:
            raise ValueError("Snapshot does not match the selected corpus.")
        if snapshot.cleaner_version != CLEANER_VERSION:
            raise ValueError(
                "Snapshot uses an older cleaner; fetch/re-clean it before "
                "tokenization.")
        audit_snapshot(
            snapshot,
            expected_section_count=spec.expected_section_count,
            expected_section_ids=_expected_section_ids(spec),
            require_no_latin=True)
        selected_config = config or BuildConfig(
            include_section_titles=(
                spec.default_include_section_titles))
        snapshot = with_section_titles(
            snapshot,
            selected_config.include_section_titles,
        )
        audit_snapshot(
            snapshot,
            expected_section_count=spec.expected_section_count,
            expected_section_ids=_expected_section_ids(spec),
            require_no_latin=True)
        selected_tokenizer = tokenizer or default_tokenizer(spec)
        selected_refiner = refiner
        if selected_refiner is _AUTOMATIC_REFINER:
            self._emit(
                "resource",
                0,
                1,
                f"Checking optional resources for {spec.title}.")
            selected_refiner = default_refiner(
                spec,
                self.corpus_root,
            )
            self._emit(
                "resource",
                1,
                1,
                f"Validated optional resources for {spec.title}.")
        def token_progress(current, total, title):
            self._emit(
                "tokenize",
                current,
                total,
                f"Tokenized {title} ({current}/{total}).")

        self._emit(
            "tokenize",
            0,
            len(snapshot.sections),
            f"Loading tokenizer for {spec.title}.")
        build = build_vocabulary(
            snapshot,
            selected_tokenizer,
            selected_config,
            progress_callback=token_progress)
        if selected_refiner is not None:
            self._emit(
                "refine",
                0,
                1,
                f"Refining long {spec.title} spans with the pinned "
                "Traditional dictionary.")
            build = refine_build_long_spans(
                build,
                selected_refiner,
            )
            self._emit(
                "refine",
                1,
                1,
                f"Refined long {spec.title} spans.")
        self._emit(
            "write",
            0,
            1,
            f"Auditing and saving {spec.title} chunks.")
        path = write_build(build, self.corpus_root)
        build = read_build(path)
        self._emit(
            "write",
            1,
            1,
            f"Saved {len(build.unique_words)} unique words.")
        return BuildResult(build, path)

    def prepare(
            self,
            spec_key,
            *,
            refresh=False,
            tokenizer=None,
            refiner=_AUTOMATIC_REFINER,
            config=None,
            client=None,
            **fetch_options):
        snapshot_result = self.fetch(
            spec_key,
            refresh=refresh,
            client=client,
            **fetch_options)
        return self.build(
            spec_key,
            snapshot=snapshot_result.snapshot,
            tokenizer=tokenizer,
            refiner=refiner,
            config=config)

    def audit_saved(self, path):
        path = Path(path)
        manifest = json.loads(
            (path / "manifest.json").read_text(encoding="utf-8"))
        kind = manifest.get("kind")
        if kind == "corpus_build":
            build = read_build(path)
            snapshot = build.snapshot
        elif kind == "corpus_snapshot":
            snapshot = read_snapshot(path)
        else:
            raise ValueError(
                "Corpus artifacts do not match the manifest kind.")
        try:
            spec = get_corpus_spec(snapshot.spec_key)
        except KeyError:
            if not _is_local_custom_snapshot(snapshot):
                raise
            # read_snapshot/read_build already performed the complete generic
            # snapshot/build audit. Repeat the public snapshot audit here to
            # make this branch explicit without imposing built-in Chinese
            # section identities or a no-Latin rule on local historical text.
            audit_snapshot(snapshot)
        else:
            audit_snapshot(
                snapshot,
                expected_section_count=spec.expected_section_count,
                expected_section_ids=_expected_section_ids(spec),
                require_no_latin=True)
        return True
