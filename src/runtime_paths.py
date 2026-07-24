"""Resolve bundled resources and per-user writable AutoAnki data."""

import shutil
import sys
from pathlib import Path

import credential_store


SOURCE_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def get_resource_root():
    """Return the source tree or PyInstaller's extracted resource tree."""
    if is_frozen():
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            return Path(bundle_root)
    return SOURCE_PROJECT_ROOT


def _get_editable_resource_directory(relative_directory, editable_name):
    bundled_directory = get_resource_root() / relative_directory
    if not is_frozen():
        return bundled_directory

    editable_directory = (
        credential_store.get_config_directory()
        / editable_name)
    editable_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if bundled_directory.is_dir():
        for bundled_path in bundled_directory.rglob("*"):
            if not bundled_path.is_file():
                continue
            relative_path = bundled_path.relative_to(bundled_directory)
            editable_path = editable_directory / relative_path
            if editable_path.exists():
                continue
            editable_path.parent.mkdir(
                mode=0o700,
                parents=True,
                exist_ok=True)
            shutil.copyfile(bundled_path, editable_path)
    return editable_directory


def get_prompt_directory():
    """Return legacy editable prompts, copying defaults on first use.

    A one-file executable is unpacked into a temporary directory each time it
    starts. Prompt edits therefore live in the user's configuration directory;
    later releases only add newly bundled prompts and never overwrite edits.
    Source checkouts continue to edit ``input/prompts`` in place.
    """
    return _get_editable_resource_directory(
        Path("input") / "prompts",
        "prompts")


def get_prompt_component_directory():
    """Return composable prompt components, persistent when frozen."""
    return _get_editable_resource_directory(
        Path("input") / "prompt_components",
        "prompt_components")


def get_output_directory():
    """Return a persistent writable output directory."""
    if is_frozen():
        return credential_store.get_config_directory() / "output"
    return SOURCE_PROJECT_ROOT / "output"


def get_words_path():
    return get_resource_root() / "input" / "words"
