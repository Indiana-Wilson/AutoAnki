"""Package and import a completed source-generation job as one Anki deck."""

import hashlib
import json
from pathlib import Path

import anki_integration
import process_text
import templates


def source_deck_name(source_title):
    title = " ".join(str(source_title).split())
    if not title:
        raise ValueError("A source title is required.")
    return f"Vocabulary from {title}"


def source_deck_id(source_key):
    source_key = str(source_key).strip()
    if not source_key:
        raise ValueError("A source identity is required.")
    digest = hashlib.sha256(
        f"autoanki-source-deck:{source_key}".encode("utf-8")
    ).digest()
    # Anki IDs are positive signed 31-bit integers. Avoid zero.
    return (int.from_bytes(digest[:4], "big") & 0x7FFFFFFF) or 1


def create_source_package(
        combined_response,
        *,
        source_title,
        source_key,
        pipeline,
        output_path,
        guid_seed=None,
        allow_accepted_content_problems=False,
        use_source_for_example_sentences=False,
        require_sentence_translations=True):
    """Validate combined structured output and write a stable `.apkg`."""
    if isinstance(combined_response, (str, bytes)):
        response_text = (
            combined_response.decode("utf-8")
            if isinstance(combined_response, bytes)
            else combined_response)
    else:
        response_text = json.dumps(
            combined_response,
            ensure_ascii=False,
            separators=(",", ":"))
    output_path = Path(output_path)
    deck_name = source_deck_name(source_title)
    deck = templates.create_deck(
        source_deck_id(source_key),
        deck_name)
    notes_created = process_text.process_json_text(
        response_text,
        deck=deck,
        output_path=output_path,
        pipeline=pipeline,
        due_start=1,
        allow_accepted_content_problems=(
            allow_accepted_content_problems),
        enforce_sentence_count=(
            not use_source_for_example_sentences),
        require_sentence_translations=require_sentence_translations,
        guid_seed=(
            guid_seed
            or f"source:{source_key}:{pipeline.pipeline_id}"))
    return output_path, notes_created


def import_source_package(
        package_path,
        *,
        client=None,
        importer=anki_integration.import_standalone_deck):
    """Import without moving cards or deleting the source-named deck."""
    return importer(
        package_path,
        client=client)
