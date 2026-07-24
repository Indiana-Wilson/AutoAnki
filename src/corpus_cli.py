"""Command-line access to AutoAnki's future corpus-import subsystem."""

import argparse
from pathlib import Path
import sys

from corpus_pipeline.catalogue import (
    get_corpus_spec,
    list_corpus_specs,
)
from corpus_pipeline.models import BuildConfig
from corpus_pipeline.service import CorpusService


def _progress(event):
    if event.total:
        progress = f" [{event.current}/{event.total}]"
    else:
        progress = ""
    print(f"{event.phase}:{progress} {event.message}", flush=True)


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch, clean, tokenize, and inspect historical Chinese works. "
            "These commands never call OpenAI or Anki."))
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the persistent corpus output directory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List built-in source works.")

    fetch = subparsers.add_parser(
        "fetch",
        help="Fetch and validate a raw Wikisource snapshot.")
    fetch.add_argument("--work", required=True)
    fetch.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch current revisions even when a snapshot is cached.")

    build = subparsers.add_parser(
        "build",
        help="Tokenize the latest saved snapshot.")
    build.add_argument("--work", required=True)
    build.add_argument("--snapshot", type=Path)
    build.add_argument("--chunk-size", type=int, default=500)
    build.add_argument(
        "--section-titles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include or exclude chapter titles (default: work-specific).")

    prepare = subparsers.add_parser(
        "prepare",
        help="Fetch (or reuse) a snapshot, then tokenize it.")
    prepare.add_argument("--work", required=True)
    prepare.add_argument("--chunk-size", type=int, default=500)
    prepare.add_argument("--refresh", action="store_true")
    prepare.add_argument(
        "--section-titles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include or exclude chapter titles (default: work-specific).")

    audit = subparsers.add_parser(
        "audit",
        help="Revalidate a serialized snapshot or build.")
    audit.add_argument("--path", type=Path, required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    service = CorpusService(
        corpus_root=arguments.output,
        progress_callback=_progress)
    if arguments.command == "list":
        for spec in list_corpus_specs():
            print(
                f"{spec.key}\t{spec.title}\t{spec.edition}\t"
                f"{spec.expected_section_count} sections")
        return 0
    if arguments.command == "fetch":
        result = service.fetch(
            arguments.work,
            refresh=arguments.refresh)
        print(result.path)
        return 0
    if arguments.command == "build":
        include_titles = (
            arguments.section_titles
            if arguments.section_titles is not None
            else get_corpus_spec(
                arguments.work
            ).default_include_section_titles
        )
        result = service.build(
            arguments.work,
            snapshot=arguments.snapshot,
            config=BuildConfig(
                chunk_size=arguments.chunk_size,
                include_section_titles=include_titles))
        print(result.path)
        return 0
    if arguments.command == "prepare":
        include_titles = (
            arguments.section_titles
            if arguments.section_titles is not None
            else get_corpus_spec(
                arguments.work
            ).default_include_section_titles
        )
        result = service.prepare(
            arguments.work,
            refresh=arguments.refresh,
            config=BuildConfig(
                chunk_size=arguments.chunk_size,
                include_section_titles=include_titles))
        print(result.path)
        return 0
    if arguments.command == "audit":
        service.audit_saved(arguments.path)
        print("Corpus audit passed.")
        return 0
    raise AssertionError("Unhandled corpus command.")


if __name__ == "__main__":
    sys.exit(main())
