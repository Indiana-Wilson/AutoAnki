"""Exact, validated translation reuse for retained source contexts.

This cache deliberately does not perform fuzzy matching, lemmatization, or
cross-language inference.  A hit requires the same NFC/LF source text and the
same exact AutoAnki language key.  Conflicting validated translations are
retained for audit but are never selected automatically.
"""

from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata

import runtime_paths


TRANSLATION_MEMORY_SCHEMA_VERSION = 1
TRANSLATION_MEMORY_POLICY = "source_context_translation_v1"
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def canonical_source_text(value):
    if not isinstance(value, str):
        raise TypeError("Translation-memory source text must be text.")
    return unicodedata.normalize(
        "NFC",
        value.replace("\r\n", "\n").replace("\r", "\n"))


def _canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"))


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def translation_memory_key(source_language_key, source_text):
    if (
            not isinstance(source_language_key, str)
            or not source_language_key.strip()):
        raise ValueError(
            "Translation memory requires an exact source language key.")
    canonical = canonical_source_text(source_text)
    material = {
        "policy": TRANSLATION_MEMORY_POLICY,
        "source_language_key": source_language_key,
        "target_language_key": "english",
        "source_text": canonical,
    }
    return "sctm1-" + _sha256_text(_canonical_json(material))


def _locally_safe_translation(value):
    return (
        isinstance(value, str)
        and bool(value.strip())
        and "|" not in html.unescape(value)
        and _HTML_TAG_PATTERN.search(value) is None)


class SourceContextTranslationMemory:
    """Small crash-safe JSON store with per-entry cross-process locks."""

    def __init__(self, root=None):
        self.root = Path(
            root
            or (
                runtime_paths.get_output_directory()
                / "source_translation_memory"
                / f"v{TRANSLATION_MEMORY_SCHEMA_VERSION}"))
        self.entries_root = self.root / "entries"
        self.locks_root = self.root / "locks"

    def _paths(self, cache_key):
        digest = cache_key.removeprefix("sctm1-")
        if (
                len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError("Invalid translation-memory key.")
        entry = self.entries_root / digest[:2] / f"{cache_key}.json"
        lock = self.locks_root / digest[:2] / f"{cache_key}.lock"
        return entry, lock

    @staticmethod
    def _atomic_write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    value,
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _read(path):
        try:
            with path.open(encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _valid_entry(
            value,
            *,
            cache_key,
            source_language_key,
            source_text):
        if (
                not isinstance(value, dict)
                or value.get("schema_version")
                != TRANSLATION_MEMORY_SCHEMA_VERSION
                or value.get("kind")
                != "source_context_translation_memory"
                or value.get("cache_key") != cache_key
                or value.get("source_language_key")
                != source_language_key
                or value.get("target_language_key") != "english"
                or value.get("source_text") != source_text
                or not isinstance(value.get("variants"), list)):
            return False
        return translation_memory_key(
            source_language_key,
            source_text) == cache_key

    def lookup(self, source_language_key, source_text):
        source_text = canonical_source_text(source_text)
        cache_key = translation_memory_key(
            source_language_key,
            source_text)
        entry_path, _lock_path = self._paths(cache_key)
        entry = self._read(entry_path)
        if not self._valid_entry(
                entry,
                cache_key=cache_key,
                source_language_key=source_language_key,
                source_text=source_text):
            return None
        variants = entry["variants"]
        if entry.get("state") != "usable" or len(variants) != 1:
            return None
        variant = variants[0]
        translation = (
            variant.get("translation")
            if isinstance(variant, dict)
            else None)
        if (
                not _locally_safe_translation(translation)
                or variant.get("translation_sha256")
                != _sha256_text(translation)):
            return None
        provenance = variant.get("provenance")
        if (
                not isinstance(provenance, list)
                or not any(
                    isinstance(item, dict)
                    and item.get("fully_validated") is True
                    and item.get("manual_acceptance") is False
                    and item.get("origin") == "provider"
                    for item in provenance)):
            return None
        return {
            "cache_key": cache_key,
            "translation": translation,
            "translation_sha256": variant["translation_sha256"],
            "entry_sha256": _sha256_text(_canonical_json(entry)),
        }

    def lookup_chunks(self, source_language_key, chunks):
        result = {}
        for chunk in chunks:
            hits = {}
            for context in chunk.contexts:
                hit = self.lookup(
                    source_language_key,
                    context.text)
                if hit is not None:
                    hits[context.context_id] = hit
            result[chunk.chunk_id] = hits
        return result

    def commit(
            self,
            source_language_key,
            source_text,
            translation,
            *,
            provenance):
        source_text = canonical_source_text(source_text)
        if not _locally_safe_translation(translation):
            raise ValueError(
                "Only nonblank plain translations can enter memory.")
        if not isinstance(provenance, dict):
            raise TypeError(
                "Translation-memory provenance must be an object.")
        if (
                provenance.get("fully_validated") is not True
                or provenance.get("manual_acceptance") is not False
                or provenance.get("origin") != "provider"):
            raise ValueError(
                "Translation memory accepts only fully validated, "
                "provider-origin translations.")
        cache_key = translation_memory_key(
            source_language_key,
            source_text)
        entry_path, lock_path = self._paths(cache_key)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            entry = self._read(entry_path)
            if not self._valid_entry(
                    entry,
                    cache_key=cache_key,
                    source_language_key=source_language_key,
                    source_text=source_text):
                entry = {
                    "schema_version": TRANSLATION_MEMORY_SCHEMA_VERSION,
                    "kind": "source_context_translation_memory",
                    "cache_key": cache_key,
                    "source_language_key": source_language_key,
                    "target_language_key": "english",
                    "source_text": source_text,
                    "state": "usable",
                    "variants": [],
                    "created_at": datetime.now(
                        timezone.utc).isoformat(),
                }
            translation_sha256 = _sha256_text(translation)
            matching = next(
                (
                    item
                    for item in entry["variants"]
                    if (
                        isinstance(item, dict)
                        and item.get("translation_sha256")
                        == translation_sha256
                        and item.get("translation") == translation)
                ),
                None)
            if matching is None:
                matching = {
                    "translation": translation,
                    "translation_sha256": translation_sha256,
                    "provenance": [],
                }
                entry["variants"].append(matching)
            provenance_id = provenance.get("provenance_id")
            if not isinstance(provenance_id, str) or not provenance_id:
                provenance_id = _sha256_text(
                    _canonical_json(provenance))
            detached_provenance = {
                **provenance,
                "provenance_id": provenance_id,
            }
            if all(
                    item.get("provenance_id") != provenance_id
                    for item in matching["provenance"]
                    if isinstance(item, dict)):
                matching["provenance"].append(detached_provenance)
            entry["state"] = (
                "usable"
                if len(entry["variants"]) == 1
                else "conflicted")
            entry["updated_at"] = datetime.now(
                timezone.utc).isoformat()
            self._atomic_write(entry_path, entry)
        return {
            "cache_key": cache_key,
            "translation_sha256": translation_sha256,
            "state": entry["state"],
            "variant_count": len(entry["variants"]),
        }
