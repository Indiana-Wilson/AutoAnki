"""Atomic, inspectable corpus snapshot and build serialization."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

from corpus_pipeline.audit import (
    audit_build,
    audit_snapshot,
    audit_vocabulary_view,
)
from corpus_pipeline.chunks import chunk_records, chunk_text, make_chunks
from corpus_pipeline.models import (
    CORPUS_PROCESSING_VERSION,
    CORPUS_SCHEMA_VERSION,
    OFFSET_CONVENTION,
    BuildConfig,
    ContextSpan,
    CorpusBuild,
    CorpusVocabularyView,
    CorpusSnapshot,
    SourcePage,
    TextSection,
    TokenOccurrence,
    TokenizerIdentity,
    UniqueWord,
    json_ready,
)


_PORTABLE_STORAGE_KEY = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
CORPUS_STORAGE_VERSION = 1


def _validate_storage_key(value, description):
    if (
            not isinstance(value, str)
            or _PORTABLE_STORAGE_KEY.fullmatch(value) is None):
        raise ValueError(
            f"{description} must be one portable lowercase storage key.")
    return value


def _validate_snapshot_storage_keys(snapshot):
    _validate_storage_key(snapshot.spec_key, "Corpus key")
    for page in snapshot.pages:
        _validate_storage_key(page.page_key, "Source page key")


def resolve_corpus_base(corpus_root, spec_key, *, create_root=False):
    """Resolve one corpus directory without following a key-level symlink."""
    _validate_storage_key(spec_key, "Corpus key")
    root = Path(corpus_root)
    if create_root:
        root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    candidate = resolved_root / spec_key
    if candidate.is_symlink():
        raise ValueError("Corpus storage key directory cannot be a symlink.")
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("Corpus storage escapes its configured root.") from error
    return resolved_candidate


def _canonical_json(value, *, indent=None):
    return json.dumps(
        json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=indent) + "\n"


def _hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
            "w",
            encoding="utf-8",
            newline="\n") as output:
        output.write(text)


def _read_text(path):
    with Path(path).open(
            "r",
            encoding="utf-8",
            newline="") as source:
        return source.read()


def _write_json(path, value):
    _write_text(path, _canonical_json(value, indent=2))


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(_canonical_json(record))


def _replace_json_pointer(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent)
    try:
        with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="\n") as output:
            output.write(_canonical_json(value, indent=2))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def snapshot_id(snapshot):
    _validate_snapshot_storage_keys(snapshot)
    identity = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "spec_key": snapshot.spec_key,
        "cleaner_version": snapshot.cleaner_version,
        "pages": [
            {
                "page_key": page.page_key,
                "revision_id": page.revision_id,
                "revision_sha1": page.revision_sha1,
                "raw_sha256": page.raw_sha256,
            }
            for page in snapshot.pages
        ],
        "canonical_sha256": _hash_text(snapshot.canonical_text),
    }
    if snapshot.include_section_titles:
        identity["include_section_titles"] = True
    return _hash_text(_canonical_json(identity))[:24]


def build_id(build):
    identity = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "processing_version": CORPUS_PROCESSING_VERSION,
        "snapshot_id": snapshot_id(build.snapshot),
        "config": build.config.to_dict(),
        "tokenizer": build.tokenizer.to_dict(),
    }
    return _hash_text(_canonical_json(identity))[:24]


def _snapshot_manifest(snapshot):
    _validate_snapshot_storage_keys(snapshot)
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "storage_version": CORPUS_STORAGE_VERSION,
        "kind": "corpus_snapshot",
        "snapshot_id": snapshot_id(snapshot),
        "created_at": _utc_now(),
        "spec_key": snapshot.spec_key,
        "edition": snapshot.edition,
        "source_language_key": snapshot.source_language_key,
        "cleaner_version": snapshot.cleaner_version,
        "include_section_titles": snapshot.include_section_titles,
        "offset_convention": OFFSET_CONVENTION,
        "canonical_sha256": _hash_text(snapshot.canonical_text),
        "counts": {
            "pages": len(snapshot.pages),
            "sections": len(snapshot.sections),
            "characters": len(snapshot.canonical_text),
        },
        "pages": [page.to_dict() for page in snapshot.pages],
    }


def _write_snapshot_artifacts(snapshot, directory):
    for page in snapshot.pages:
        _write_text(
            directory / "source" / f"{page.page_key}.wiki",
            page.raw_wikitext)
    _write_text(
        directory / "clean" / "text.txt",
        snapshot.canonical_text)
    _write_jsonl(
        directory / "clean" / "sections.jsonl",
        snapshot.sections)
    manifest = _snapshot_manifest(snapshot)
    manifest["artifacts"] = _artifact_hashes(directory)
    _write_json(directory / "manifest.json", manifest)


def _publish_directory(parent, target_name, writer, validator):
    parent = Path(parent)
    if parent.is_symlink():
        raise ValueError(
            "Corpus artifact parent directory cannot be a symlink.")
    resolved_container = parent.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    if (
            parent.is_symlink()
            or parent.resolve()
            != resolved_container / parent.name):
        raise ValueError(
            "Corpus artifact parent directory escapes its corpus.")
    target = parent / target_name
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise ValueError(
                f"Corpus artifact target is not a safe directory: {target}")
        validator(target)
        return target
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{target_name}.",
        dir=parent))
    try:
        writer(temporary)
        validator(temporary)
        try:
            temporary.replace(target)
        except OSError:
            if target.is_symlink() or not target.is_dir():
                raise
            validator(target)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def write_snapshot(
        snapshot,
        corpus_root,
        *,
        expected_section_count=None,
        expected_section_ids=None,
        require_no_latin=False):
    _validate_snapshot_storage_keys(snapshot)
    audit_snapshot(
        snapshot,
        expected_section_count=expected_section_count,
        expected_section_ids=expected_section_ids,
        require_no_latin=require_no_latin)
    corpus_root = Path(corpus_root).resolve()
    corpus_base = resolve_corpus_base(
        corpus_root,
        snapshot.spec_key,
        create_root=True)
    identifier = snapshot_id(snapshot)
    parent = corpus_base / "snapshots"
    target = _publish_directory(
        parent,
        identifier,
        lambda temporary: _write_snapshot_artifacts(
            snapshot,
            temporary),
        lambda directory: _validate_snapshot_directory(
            directory,
            snapshot))
    _replace_json_pointer(
        corpus_base / "latest_snapshot.json",
        {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "snapshot_id": identifier,
            "relative_path": target.relative_to(
                corpus_base).as_posix(),
        })
    return target


def _read_snapshot(path, *, artifacts_verified):
    path = Path(path)
    manifest = json.loads(
        _read_text(path / "manifest.json"))
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("Unsupported corpus snapshot schema version.")
    if manifest.get("kind") not in {
            "corpus_snapshot",
            "corpus_build"}:
        raise ValueError("Unsupported corpus manifest kind.")
    _validate_storage_key(manifest.get("spec_key"), "Corpus key")
    if manifest.get("offset_convention") != OFFSET_CONVENTION:
        raise ValueError("Corpus offset convention is inconsistent.")
    storage_version = manifest.get("storage_version")
    if storage_version not in {None, CORPUS_STORAGE_VERSION}:
        raise ValueError("Unsupported corpus storage version.")
    artifact_map = manifest.get("artifacts")
    legacy_snapshot = (
        artifact_map is None
        and manifest.get("kind") == "corpus_snapshot"
        and storage_version is None
    )
    if not isinstance(artifact_map, dict) and not legacy_snapshot:
        raise ValueError("Corpus manifest has no valid artifact map.")
    if not artifacts_verified and not legacy_snapshot:
        verify_artifact_hashes(path)
        if artifact_map != _artifact_hashes(path):
            raise ValueError(
                "Corpus directory contains an unrecorded or mismatched "
                "artifact.")
    pages = []
    for metadata in manifest["pages"]:
        page_key = metadata.get("page_key")
        _validate_storage_key(page_key, "Source page key")
        raw_wikitext = _read_text(
            path
            / "source"
            / f"{page_key}.wiki")
        pages.append(SourcePage(
            **metadata,
            raw_wikitext=raw_wikitext))
    sections = []
    with (path / "clean" / "sections.jsonl").open(
            encoding="utf-8") as source:
        for line in source:
            if line.strip():
                sections.append(TextSection(**json.loads(line)))
    include_section_titles = manifest.get(
        "include_section_titles",
        False,
    )
    if not isinstance(include_section_titles, bool):
        raise ValueError(
            "Corpus title-inclusion metadata must be true or false.")
    snapshot = CorpusSnapshot(
        spec_key=manifest["spec_key"],
        edition=manifest["edition"],
        source_language_key=manifest["source_language_key"],
        pages=tuple(pages),
        sections=tuple(sections),
        canonical_text=_read_text(path / "clean" / "text.txt"),
        cleaner_version=manifest["cleaner_version"],
        include_section_titles=include_section_titles)
    audit_snapshot(snapshot)
    if manifest.get("canonical_sha256") != _hash_text(
            snapshot.canonical_text):
        raise ValueError("Corpus canonical-text hash is inconsistent.")
    if snapshot_id(snapshot) != manifest["snapshot_id"]:
        raise ValueError("Corpus snapshot identity check failed.")
    if manifest.get("kind") == "corpus_snapshot":
        expected_counts = {
            "pages": len(snapshot.pages),
            "sections": len(snapshot.sections),
            "characters": len(snapshot.canonical_text),
        }
        if manifest.get("counts") != expected_counts:
            raise ValueError(
                "Corpus snapshot count metadata is inconsistent.")
    return snapshot


def read_snapshot(path):
    """Deserialize a snapshot, verifying all recorded directory artifacts."""
    return _read_snapshot(path, artifacts_verified=False)


def _page_content_identity(page):
    identity = page.to_dict(include_wikitext=True)
    identity.pop("retrieved_at")
    return identity


def _snapshots_equivalent(left, right):
    """Compare immutable source content, ignoring only retrieval wall time."""
    return (
        left.spec_key == right.spec_key
        and left.edition == right.edition
        and left.source_language_key == right.source_language_key
        and left.cleaner_version == right.cleaner_version
        and left.include_section_titles == right.include_section_titles
        and left.canonical_text == right.canonical_text
        and left.sections == right.sections
        and tuple(
            _page_content_identity(page)
            for page in left.pages
        ) == tuple(
            _page_content_identity(page)
            for page in right.pages
        )
    )


def _validate_snapshot_directory(path, expected_snapshot):
    manifest = json.loads(_read_text(Path(path) / "manifest.json"))
    if manifest.get("kind") != "corpus_snapshot":
        raise ValueError(
            "An existing snapshot target has the wrong manifest kind.")
    if not isinstance(manifest.get("artifacts"), dict):
        raise ValueError(
            "An existing snapshot target predates validated atomic reuse.")
    if manifest.get("snapshot_id") != snapshot_id(expected_snapshot):
        raise ValueError(
            "An existing snapshot target has the wrong identity.")
    loaded = read_snapshot(path)
    if not _snapshots_equivalent(loaded, expected_snapshot):
        raise ValueError(
            "An existing snapshot target disagrees with current content.")
    return loaded


def verify_artifact_hashes(path):
    """Verify every artifact recorded by a serialized corpus manifest."""
    path = Path(path)
    manifest = json.loads(
        _read_text(path / "manifest.json"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("This manifest does not contain corpus artifacts.")
    for relative_name, expected_hash in artifacts.items():
        if not isinstance(relative_name, str) or "\\" in relative_name:
            raise ValueError(
                "Corpus manifest contains a non-portable artifact path.")
        portable_path = PurePosixPath(relative_name)
        if portable_path.is_absolute() or ".." in portable_path.parts:
            raise ValueError(
                "Corpus manifest contains an unsafe artifact path.")
        artifact = path.joinpath(*portable_path.parts).resolve()
        try:
            artifact.relative_to(path.resolve())
        except ValueError as error:
            raise ValueError(
                "Corpus manifest contains an unsafe artifact path.") from error
        if not artifact.is_file():
            raise ValueError(
                f"Corpus artifact is missing: {relative_name}")
        if _hash_file(artifact) != expected_hash:
            raise ValueError(
                f"Corpus artifact hash failed: {relative_name}")
    return True


def _read_jsonl(path, record_type):
    records = []
    with Path(path).open(
            "r",
            encoding="utf-8",
            newline="") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(record_type(**json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid {Path(path).name} record on line "
                    f"{line_number}.") from error
    return tuple(records)


def read_build(path):
    """Deserialize and semantically audit a complete saved corpus build."""
    path = Path(path)
    manifest = json.loads(_read_text(path / "manifest.json"))
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("Unsupported corpus build schema version.")
    if manifest.get("kind") != "corpus_build":
        raise ValueError("This directory is not a corpus build.")
    if (
            manifest.get("processing_version")
            != CORPUS_PROCESSING_VERSION):
        raise ValueError(
            "Saved corpus build uses another processing version.")
    verify_artifact_hashes(path)
    if manifest["artifacts"] != _artifact_hashes(path):
        raise ValueError(
            "Corpus build contains an unrecorded or mismatched artifact.")

    snapshot = _read_snapshot(path, artifacts_verified=True)
    config_data = manifest.get("config")
    tokenizer_data = manifest.get("tokenizer")
    if not isinstance(config_data, dict):
        raise ValueError("Corpus build has no valid configuration.")
    if not isinstance(tokenizer_data, dict):
        raise ValueError("Corpus build has no valid tokenizer identity.")
    tokenizer_data = dict(tokenizer_data)
    options = tokenizer_data.pop("options", {})
    if not isinstance(options, dict):
        raise ValueError("Corpus tokenizer options must be an object.")
    try:
        config = BuildConfig(**config_data)
        tokenizer = TokenizerIdentity(
            **tokenizer_data,
            options=tuple(sorted(
                (str(key), str(value))
                for key, value in options.items())))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Corpus build metadata cannot be deserialized.") from error

    build = CorpusBuild(
        snapshot=snapshot,
        config=config,
        tokenizer=tokenizer,
        contexts=_read_jsonl(
            path / "contexts.jsonl",
            ContextSpan),
        occurrences=_read_jsonl(
            path / "occurrences.jsonl",
            TokenOccurrence),
        unique_words=_read_jsonl(
            path / "unique_words.jsonl",
            UniqueWord))
    audit_build(build)

    expected_identifier = build_id(build)
    if manifest.get("build_id") != expected_identifier:
        raise ValueError("Corpus build identity check failed.")
    chunks = make_chunks(
        build.unique_words,
        build.config.chunk_size)
    expected_chunk_names = {
        f"{chunk.stem}.{suffix}"
        for chunk in chunks
        for suffix in ("json", "txt")
    }
    actual_chunk_names = {
        chunk_path.name
        for chunk_path in (path / "chunks").iterdir()
        if chunk_path.is_file()
    } if (path / "chunks").is_dir() else set()
    if actual_chunk_names != expected_chunk_names:
        raise ValueError("Corpus chunk files do not cover the saved ranks.")
    for chunk in chunks:
        if _read_text(
                path / "chunks" / f"{chunk.stem}.txt") != chunk_text(chunk):
            raise ValueError(
                f"Corpus text chunk is inconsistent: {chunk.stem}")
        expected_json = {
            "start_rank": chunk.start_rank,
            "end_rank": chunk.end_rank,
            "words": chunk_records(build, chunk),
        }
        actual_json = json.loads(_read_text(
            path / "chunks" / f"{chunk.stem}.json"))
        if actual_json != expected_json:
            raise ValueError(
                f"Corpus JSON chunk is inconsistent: {chunk.stem}")
    expected_counts = {
        "pages": len(build.snapshot.pages),
        "sections": len(build.snapshot.sections),
        "characters": len(build.snapshot.canonical_text),
        "contexts": len(build.contexts),
        "token_occurrences": len(build.occurrences),
        "unique_words": len(build.unique_words),
        "chunks": len(chunks),
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError("Corpus build count metadata is inconsistent.")
    report = json.loads(_read_text(path / "audit" / "report.json"))
    if (
            report.get("status") != "passed"
            or report.get("snapshot_id") != snapshot_id(snapshot)
            or report.get("build_id") != expected_identifier
            or report.get("processing_version")
            != CORPUS_PROCESSING_VERSION
            or report.get("counts") != expected_counts):
        raise ValueError("Corpus build audit report is inconsistent.")
    return build


def read_vocabulary_view(path):
    """Read and audit generation data without parsing all occurrences."""
    path = Path(path)
    manifest = json.loads(_read_text(path / "manifest.json"))
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("Unsupported corpus build schema version.")
    if manifest.get("kind") != "corpus_build":
        raise ValueError("This directory is not a corpus build.")
    if (
            manifest.get("processing_version")
            != CORPUS_PROCESSING_VERSION):
        raise ValueError(
            "Saved corpus build uses another processing version.")
    verify_artifact_hashes(path)
    actual_artifacts = {
        artifact.relative_to(path).as_posix()
        for artifact in path.rglob("*")
        if artifact.is_file() and artifact.name != "manifest.json"
    }
    if actual_artifacts != set(manifest["artifacts"]):
        raise ValueError(
            "Corpus build contains an unrecorded or missing artifact.")

    snapshot = _read_snapshot(path, artifacts_verified=True)
    config_data = manifest.get("config")
    tokenizer_data = manifest.get("tokenizer")
    if not isinstance(config_data, dict):
        raise ValueError("Corpus build has no valid configuration.")
    if not isinstance(tokenizer_data, dict):
        raise ValueError("Corpus build has no valid tokenizer identity.")
    tokenizer_data = dict(tokenizer_data)
    options = tokenizer_data.pop("options", {})
    if not isinstance(options, dict):
        raise ValueError("Corpus tokenizer options must be an object.")
    try:
        config = BuildConfig(**config_data)
        tokenizer = TokenizerIdentity(
            **tokenizer_data,
            options=tuple(sorted(
                (str(key), str(value))
                for key, value in options.items())))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Corpus build metadata cannot be deserialized.") from error

    view = CorpusVocabularyView(
        snapshot=snapshot,
        config=config,
        tokenizer=tokenizer,
        contexts=_read_jsonl(
            path / "contexts.jsonl",
            ContextSpan),
        unique_words=_read_jsonl(
            path / "unique_words.jsonl",
            UniqueWord))
    audit_vocabulary_view(view)
    if manifest.get("build_id") != build_id(view):
        raise ValueError("Corpus build identity check failed.")
    counts = manifest.get("counts", {})
    if (
            counts.get("pages") != len(view.snapshot.pages)
            or counts.get("sections") != len(view.snapshot.sections)
            or counts.get("characters") != len(
                view.snapshot.canonical_text)
            or counts.get("contexts") != len(view.contexts)
            or counts.get("unique_words") != len(view.unique_words)
            or counts.get("chunks") != len(make_chunks(
                view.unique_words,
                view.config.chunk_size))):
        raise ValueError("Corpus build count metadata is inconsistent.")
    report = json.loads(_read_text(path / "audit" / "report.json"))
    if (
            report.get("status") != "passed"
            or report.get("build_id") != manifest["build_id"]
            or report.get("processing_version")
            != CORPUS_PROCESSING_VERSION
            or report.get("counts") != counts):
        raise ValueError("Corpus build audit report is inconsistent.")
    return view


def _validate_build_directory(path, expected_build):
    loaded = read_build(path)
    if (
            not _snapshots_equivalent(
                loaded.snapshot,
                expected_build.snapshot)
            or loaded.config != expected_build.config
            or loaded.tokenizer.to_dict()
            != expected_build.tokenizer.to_dict()
            or loaded.contexts != expected_build.contexts
            or loaded.occurrences != expected_build.occurrences
            or loaded.unique_words != expected_build.unique_words):
        raise ValueError(
            "An existing build target disagrees with current output.")
    return loaded


def _safe_tsv(value):
    return str(value).replace("\t", " ").replace("\r", " ").replace(
        "\n",
        " ")


def _write_boundaries(build, path):
    lines = [
        "order\tsection_id\ttitle\tsource_page\tcharacters"
        "\tfirst_40_characters\tlast_40_characters",
    ]
    for section in build.snapshot.sections:
        lines.append("\t".join(_safe_tsv(value) for value in (
            section.order,
            section.section_id,
            section.title,
            section.source_page_key,
            len(section.text),
            section.text[:40],
            section.text[-40:],
        )))
    _write_text(path, "\n".join(lines) + "\n")


def _artifact_hashes(directory):
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            hashes[path.relative_to(directory).as_posix()] = _hash_file(path)
    return hashes


def _write_build_artifacts(build, directory):
    _write_snapshot_artifacts(build.snapshot, directory)
    _write_jsonl(directory / "contexts.jsonl", build.contexts)
    _write_jsonl(
        directory / "occurrences.jsonl",
        build.occurrences)
    _write_jsonl(
        directory / "unique_words.jsonl",
        build.unique_words)

    chunks = make_chunks(
        build.unique_words,
        build.config.chunk_size)
    for chunk in chunks:
        _write_text(
            directory / "chunks" / f"{chunk.stem}.txt",
            chunk_text(chunk))
        _write_json(
            directory / "chunks" / f"{chunk.stem}.json",
            {
                "start_rank": chunk.start_rank,
                "end_rank": chunk.end_rank,
                "words": chunk_records(build, chunk),
            })

    _write_boundaries(
        build,
        directory / "audit" / "chapter_boundaries.tsv")
    audit_report = {
        "status": "passed",
        "snapshot_id": snapshot_id(build.snapshot),
        "build_id": build_id(build),
        "processing_version": CORPUS_PROCESSING_VERSION,
        "counts": {
            "pages": len(build.snapshot.pages),
            "sections": len(build.snapshot.sections),
            "characters": len(build.snapshot.canonical_text),
            "contexts": len(build.contexts),
            "token_occurrences": len(build.occurrences),
            "unique_words": len(build.unique_words),
            "chunks": len(chunks),
        },
        "checks": (
            "source completeness",
            "clean-text contamination",
            "source/context/token offsets",
            "first-occurrence ranking",
            "chunk coverage",
        ),
    }
    _write_json(directory / "audit" / "report.json", audit_report)
    _write_text(
        directory / "audit" / "report.txt",
        "\n".join((
            "Corpus audit: PASSED",
            f"Sections: {len(build.snapshot.sections)}",
            f"Characters: {len(build.snapshot.canonical_text)}",
            f"Token occurrences: {len(build.occurrences)}",
            f"Unique words: {len(build.unique_words)}",
            f"Chunks: {len(chunks)}",
            "",
            "No OpenAI or Anki operation is part of this build.",
            "",
        )))

    manifest = _snapshot_manifest(build.snapshot)
    manifest.update({
        "kind": "corpus_build",
        "build_id": build_id(build),
        "snapshot_id": snapshot_id(build.snapshot),
        "processing_version": CORPUS_PROCESSING_VERSION,
        "config": build.config.to_dict(),
        "tokenizer": build.tokenizer.to_dict(),
        "counts": audit_report["counts"],
        "artifacts": _artifact_hashes(directory),
    })
    _write_json(directory / "manifest.json", manifest)


def write_build(build, corpus_root):
    _validate_snapshot_storage_keys(build.snapshot)
    audit_snapshot(build.snapshot)
    audit_build(build)
    corpus_root = Path(corpus_root).resolve()
    corpus_base = resolve_corpus_base(
        corpus_root,
        build.snapshot.spec_key,
        create_root=True)
    identifier = build_id(build)
    parent = corpus_base / "runs"
    target = _publish_directory(
        parent,
        identifier,
        lambda temporary: _write_build_artifacts(
            build,
            temporary),
        lambda directory: _validate_build_directory(
            directory,
            build))
    _replace_json_pointer(
        corpus_base / "latest_build.json",
        {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "build_id": identifier,
            "relative_path": target.relative_to(
                corpus_base).as_posix(),
        })
    return target
