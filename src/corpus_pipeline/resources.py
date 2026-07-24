"""Pinned, verified optional resources used by corpus processing."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen


JIEBA_TRADITIONAL_REVISION = (
    "67fa2e36e72f69d9134b8a1037b83fbb070b9775"
)
JIEBA_TRADITIONAL_URL = (
    "https://raw.githubusercontent.com/fxsjy/jieba/"
    f"{JIEBA_TRADITIONAL_REVISION}/extra_dict/dict.txt.big"
)
JIEBA_TRADITIONAL_SHA256 = (
    "b16011275c42955ccd81fc1adecc93a59"
    "dbb7926af69d93fc95d4943d40f6aad"
)
JIEBA_TRADITIONAL_LICENSE = "MIT"
_MAX_DICTIONARY_BYTES = 20 * 1024 * 1024


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_dictionary(url):
    request = Request(
        url,
        headers={
            "User-Agent": (
                "AutoAnki corpus preparation/1.0 "
                "(pinned Jieba dictionary resource)"
            ),
        },
    )
    with urlopen(request, timeout=60) as response:
        data = response.read(_MAX_DICTIONARY_BYTES + 1)
    if len(data) > _MAX_DICTIONARY_BYTES:
        raise ValueError("The dictionary download exceeded its size limit.")
    return data


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _resource_parent(corpus_root):
    root = Path(corpus_root)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    resources = root / "_resources"
    if resources.is_symlink():
        raise ValueError("Corpus resource storage cannot be a symlink.")
    resources.mkdir(exist_ok=True)
    parent = resources / "jieba_traditional"
    if parent.is_symlink():
        raise ValueError("Corpus resource storage cannot be a symlink.")
    parent.mkdir(exist_ok=True)
    resolved_parent = parent.resolve()
    try:
        resolved_parent.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "Corpus resource storage escapes its configured root.") from error
    return resolved_parent


def ensure_jieba_traditional_dictionary(
        corpus_root,
        *,
        download_bytes=None,
        expected_sha256=JIEBA_TRADITIONAL_SHA256,
        source_url=JIEBA_TRADITIONAL_URL,
        source_revision=JIEBA_TRADITIONAL_REVISION):
    """Return a verified, content-addressed Traditional Jieba dictionary."""
    if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None):
        raise ValueError(
            "The expected dictionary SHA-256 must be 64 lowercase "
            "hexadecimal characters.")
    resource_parent = _resource_parent(corpus_root)
    resource_directory = resource_parent / expected_sha256[:24]
    if resource_directory.is_symlink():
        raise ValueError("Corpus resource version cannot be a symlink.")
    resource_directory.mkdir(exist_ok=True)
    resolved_directory = resource_directory.resolve()
    try:
        resolved_directory.relative_to(resource_parent)
    except ValueError as error:
        raise ValueError(
            "Corpus resource version escapes resource storage.") from error

    dictionary_path = resolved_directory / "dict.txt.big"
    manifest_path = resolved_directory / "manifest.json"
    if dictionary_path.exists():
        if (
                not dictionary_path.is_file()
                or dictionary_path.is_symlink()
                or sha256_file(dictionary_path) != expected_sha256):
            raise ValueError(
                "The cached Traditional Jieba dictionary failed its hash.")
    else:
        fetch = download_bytes or _download_dictionary
        data = fetch(source_url)
        if not isinstance(data, bytes):
            raise TypeError("Dictionary downloader must return bytes.")
        if len(data) > _MAX_DICTIONARY_BYTES:
            raise ValueError(
                "The dictionary download exceeded its size limit.")
        if _sha256_bytes(data) != expected_sha256:
            raise ValueError(
                "The Traditional Jieba dictionary download failed its hash.")
        _atomic_write(dictionary_path, data)

    manifest = {
        "kind": "corpus_dictionary_resource",
        "name": "jieba Traditional Chinese large dictionary",
        "source_url": source_url,
        "source_revision": source_revision,
        "sha256": expected_sha256,
        "bytes": dictionary_path.stat().st_size,
        "license": JIEBA_TRADITIONAL_LICENSE,
    }
    _atomic_write(
        manifest_path,
        (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ) + "\n"
        ).encode("utf-8"),
    )
    return dictionary_path
