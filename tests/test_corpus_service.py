import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import corpus_cli
from corpus_pipeline import catalogue, mediawiki
from corpus_pipeline.cleaners import CLEANER_VERSION
from corpus_pipeline.models import (
    CORPUS_SCHEMA_VERSION,
    BuildConfig,
    CorpusValidationError,
    TokenSpan,
    TokenizerIdentity,
)
from corpus_pipeline.service import (
    CorpusService,
    ProgressEvent,
    default_refiner,
    make_snapshot,
)
from corpus_pipeline.storage import write_snapshot
from corpus_pipeline.tokenizers import CkipHanTokenizer


CHINESE_NUMERALS = tuple(
    (
        "一 二 三 四 五 六 七 八 九 十 十一 十二 十三 十四 十五 "
        "十六 十七 十八 十九 二十 二十一 二十二 二十三 二十四 "
        "二十五 二十六 二十七 二十八 二十九 三十 三十一 三十二 "
        "三十三 三十四 三十五 三十六 三十七 三十八 三十九 四十 "
        "四十一 四十二 四十三 四十四 四十五 四十六 四十七 四十八 "
        "四十九 五十 五十一 五十二 五十三 五十四 五十五 五十六 "
        "五十七 五十八 五十九 六十 六十一 六十二 六十三 六十四 "
        "六十五 六十六 六十七 六十八 六十九 七十 七十一 七十二 "
        "七十三 七十四 七十五 七十六 七十七 七十八 七十九 八十 "
        "八十一"
    ).split()
)


def _daodejing_wikitext(first_chapter="道德。"):
    chapters = []
    for number, numeral in enumerate(CHINESE_NUMERALS, start=1):
        body = (
            "道可道，非常道。" + first_chapter
            if number == 1
            else "天地。")
        chapters.append(f"=={numeral}章==\n{body}\n")
    return (
        "{{header|title=老子}}\n"
        "== 道經 ==\n"
        + "".join(chapters)
        + "{{footer}}\n"
        "[[Category:道經]]\n"
    )


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeMediaWikiClient:
    """MediaWiki-compatible fixture with no network access."""

    def __init__(self, content_by_title, *, revision_id=1001):
        self.content_by_title = dict(content_by_title)
        self.revision_id = revision_id
        self.calls = []

    def get(self, url, *, params, headers, timeout):
        self.calls.append({
            "url": url,
            "params": params,
            "headers": headers,
            "timeout": timeout,
        })
        titles = params["titles"].split("|")
        pages = []
        for title in reversed(titles):
            raw = self.content_by_title[title]
            pages.append({
                "title": title,
                "fullurl": (
                    "https://zh.wikisource.org/wiki/"
                    + title.replace(" ", "_")
                ),
                "revisions": [{
                    "revid": self.revision_id,
                    "timestamp": "2026-07-24T00:00:00Z",
                    "sha1": hashlib.sha1(
                        raw.encode("utf-8")
                    ).hexdigest(),
                    "slots": {
                        "main": {
                            "content": raw,
                        },
                    },
                }],
            })
        return FakeResponse({"query": {"pages": pages}})


class NeverCalledClient:
    def __init__(self):
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("A cached fetch must not contact MediaWiki.")


class FakeTokenizer:
    """Small deterministic tokenizer that cannot load a remote model."""

    @property
    def identity(self):
        return TokenizerIdentity(
            backend="test-character",
            backend_version="1",
            model="offline-code-points",
            model_revision="fixture-1",
        )

    def tokenize(self, text):
        return tuple(
            TokenSpan(
                surface=character,
                start_offset=index,
                end_offset=index + 1,
                confidence=1.0,
            )
            for index, character in enumerate(text)
            if not character.isspace()
        )


def _client(first_chapter="道德。", *, revision_id=1001):
    title = catalogue.DAODEJING.content_titles[0]
    return FakeMediaWikiClient(
        {title: _daodejing_wikitext(first_chapter)},
        revision_id=revision_id,
    )


def _fetched_pages(client=None):
    return mediawiki.fetch_corpus_pages(
        catalogue.DAODEJING,
        client=client or _client(),
        retrieved_at="2026-07-24T01:02:03Z",
    )


class SnapshotServiceTests(unittest.TestCase):
    def test_make_snapshot_cleans_all_sections_and_aligns_offsets(self):
        pages = _fetched_pages()

        snapshot = make_snapshot(catalogue.DAODEJING, pages)

        self.assertEqual(snapshot.spec_key, catalogue.DAODEJING.key)
        self.assertEqual(snapshot.pages, pages)
        self.assertEqual(len(snapshot.sections), 81)
        self.assertEqual(
            snapshot.sections[0].section_id,
            "daodejing_wang_bi:chapter:001",
        )
        self.assertEqual(
            snapshot.sections[-1].section_id,
            "daodejing_wang_bi:chapter:081",
        )
        self.assertEqual(
            snapshot.sections[0].text,
            "道可道，非常道。道德。")
        for section in snapshot.sections:
            self.assertEqual(
                snapshot.canonical_text[
                    section.start_offset:section.end_offset
                ],
                section.text,
            )

    def test_make_snapshot_accepts_a_single_pass_page_iterable(self):
        pages = _fetched_pages()

        snapshot = make_snapshot(catalogue.DAODEJING, iter(pages))

        self.assertEqual(snapshot.pages, pages)
        self.assertEqual(len(snapshot.sections), 81)

    def test_fetch_reuses_cache_and_refreshes_only_when_requested(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = CorpusService(root, progress_callback=events.append)
            initial_client = _client()

            first = service.fetch(
                catalogue.DAODEJING.key,
                client=initial_client,
                retrieved_at="2026-07-24T01:02:03Z",
            )
            cached_client = NeverCalledClient()
            cached = service.fetch(
                catalogue.DAODEJING.key,
                client=cached_client,
            )
            refreshed_client = _client(
                "玄德。",
                revision_id=2002,
            )
            refreshed = service.fetch(
                catalogue.DAODEJING.key,
                refresh=True,
                client=refreshed_client,
                retrieved_at="2026-07-24T02:03:04Z",
            )

            self.assertFalse(first.from_cache)
            self.assertTrue(cached.from_cache)
            self.assertFalse(refreshed.from_cache)
            self.assertEqual(cached.path, first.path)
            self.assertNotEqual(refreshed.path, first.path)
            self.assertEqual(cached_client.calls, 0)
            self.assertEqual(len(initial_client.calls), 1)
            self.assertEqual(len(refreshed_client.calls), 1)
            pointer = json.loads(
                (
                    root
                    / catalogue.DAODEJING.key
                    / "latest_snapshot.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(pointer["snapshot_id"], refreshed.path.name)
            self.assertEqual(
                (root / catalogue.DAODEJING.key / pointer["relative_path"]),
                refreshed.path,
            )

        self.assertTrue(all(
            isinstance(event, ProgressEvent)
            for event in events
        ))
        self.assertIn(
            ("fetch", 1, 1, "Using saved 道德經 source snapshot."),
            tuple(
                (
                    event.phase,
                    event.current,
                    event.total,
                    event.message,
                )
                for event in events
            ),
        )

    def test_failed_refresh_leaves_the_previous_cache_pointer_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = CorpusService(root)
            initial = service.fetch(
                catalogue.DAODEJING.key,
                client=_client(),
                retrieved_at="2026-07-24T01:02:03Z",
            )
            pointer_path = (
                root
                / catalogue.DAODEJING.key
                / "latest_snapshot.json"
            )
            original_pointer = pointer_path.read_bytes()
            malformed = FakeMediaWikiClient(
                {
                    catalogue.DAODEJING.content_titles[0]:
                        "=== 一章 ===\n只有一章。",
                },
                revision_id=2002,
            )

            with self.assertRaises(CorpusValidationError):
                service.fetch(
                    catalogue.DAODEJING.key,
                    refresh=True,
                    client=malformed,
                )

            self.assertEqual(pointer_path.read_bytes(), original_pointer)
            self.assertEqual(
                service.load_latest_snapshot(
                    catalogue.DAODEJING.key
                ).path,
                initial.path,
            )

    def test_cached_raw_source_is_recleaned_after_cleaner_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = make_snapshot(
                catalogue.DAODEJING,
                _fetched_pages(_client("其事好还。")))
            stale = replace(
                current,
                cleaner_version="obsolete-cleaner")
            stale_path = write_snapshot(
                stale,
                root,
                expected_section_count=81,
                expected_section_ids=tuple(
                    f"daodejing_wang_bi:chapter:{number:03d}"
                    for number in range(1, 82)),
                require_no_latin=True)
            client = NeverCalledClient()

            result = CorpusService(root).fetch(
                catalogue.DAODEJING.key,
                client=client)

            self.assertTrue(result.from_cache)
            self.assertEqual(client.calls, 0)
            self.assertNotEqual(result.path, stale_path)
            self.assertEqual(
                result.snapshot.cleaner_version,
                CLEANER_VERSION)
            self.assertEqual(
                result.snapshot.sections[0].text,
                "道可道，非常道。其事好还。")


class BuildServiceTests(unittest.TestCase):
    def test_builtin_corpus_holds_one_ckip_session_for_all_sections(self):
        events = []
        active = []
        tokenizer = CkipHanTokenizer(
            "fixture/model",
            "fixture-revision",
            device="cuda")
        tokenizer._resolved_device = "cuda"

        @contextmanager
        def execution_session():
            events.append("lease-acquired")
            active.append(True)
            try:
                yield tokenizer
            finally:
                active.pop()
                events.append("lease-released")

        def tokenize(text):
            self.assertTrue(active)
            events.append("section-tokenized")
            return tuple(
                TokenSpan(character, index, index + 1, 0.75)
                for index, character in enumerate(text)
                if not character.isspace())

        tokenizer.execution_session = execution_session
        tokenizer.tokenize = tokenize

        with tempfile.TemporaryDirectory() as directory:
            service = CorpusService(directory)
            fetched = service.fetch(
                catalogue.DAODEJING.key,
                client=_client())
            service.build(
                catalogue.DAODEJING.key,
                snapshot=fetched.snapshot,
                tokenizer=tokenizer,
                refiner=None)

        self.assertEqual(events[0], "lease-acquired")
        self.assertEqual(events[-1], "lease-released")
        self.assertEqual(events.count("section-tokenized"), 81)

    def test_build_writes_pointer_reports_progress_and_audits_offline(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = CorpusService(root, progress_callback=events.append)
            fetched = service.fetch(
                catalogue.DAODEJING.key,
                client=_client(),
                retrieved_at="2026-07-24T01:02:03Z",
            )
            events.clear()

            result = service.build(
                catalogue.DAODEJING.key,
                snapshot=fetched.path,
                tokenizer=FakeTokenizer(),
                config=BuildConfig(chunk_size=2),
            )

            self.assertEqual(result.build.snapshot, fetched.snapshot)
            self.assertEqual(
                tuple(
                    word.surface
                    for word in result.build.unique_words
                ),
                ("道", "可", "非", "常", "德", "天", "地"),
            )
            self.assertTrue(result.path.is_dir())
            pointer = json.loads(
                (
                    root
                    / catalogue.DAODEJING.key
                    / "latest_build.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(pointer["build_id"], result.path.name)
            self.assertEqual(
                root
                / catalogue.DAODEJING.key
                / pointer["relative_path"],
                result.path,
            )
            self.assertTrue(service.audit_saved(fetched.path))
            self.assertTrue(service.audit_saved(result.path))

            token_events = [
                event for event in events
                if event.phase == "tokenize"
            ]
            write_events = [
                event for event in events
                if event.phase == "write"
            ]
            self.assertEqual(
                (token_events[0].current, token_events[0].total),
                (0, 81),
            )
            self.assertEqual(
                (token_events[-1].current, token_events[-1].total),
                (81, 81),
            )
            self.assertEqual(
                tuple((event.current, event.total) for event in write_events),
                ((0, 1), (1, 1)),
            )
        self.assertIn(
            "Saved 7 unique words.",
                write_events[-1].message,
            )

    def test_refinement_policy_is_automatic_but_can_be_disabled_explicitly(
            self):
        snapshot = SimpleNamespace(
            spec_key=catalogue.JOURNEY_TO_THE_WEST.key,
            cleaner_version=CLEANER_VERSION,
            sections=(),
            include_section_titles=False,
        )
        custom_tokenizer = object()
        automatic_refiner = object()
        unrefined = SimpleNamespace(unique_words=())
        refined = SimpleNamespace(unique_words=())

        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory) / "run"
            events = []
            service = CorpusService(
                directory,
                progress_callback=events.append,
            )
            with (
                    patch("corpus_pipeline.service.audit_snapshot"),
                    patch(
                        "corpus_pipeline.service.default_refiner",
                        return_value=automatic_refiner,
                    ) as choose_refiner,
                    patch(
                        "corpus_pipeline.service.build_vocabulary",
                        return_value=unrefined,
                    ),
                    patch(
                        "corpus_pipeline.service.refine_build_long_spans",
                        return_value=refined,
                    ) as apply_refiner,
                    patch(
                        "corpus_pipeline.service.write_build",
                        return_value=run_path,
                    ) as write_build,
                    patch(
                        "corpus_pipeline.service.read_build",
                        side_effect=(refined, unrefined),
                    )):
                service.build(
                    catalogue.JOURNEY_TO_THE_WEST.key,
                    snapshot=snapshot,
                    tokenizer=custom_tokenizer,
                    config=BuildConfig(
                        include_section_titles=False),
                )
                service.build(
                    catalogue.JOURNEY_TO_THE_WEST.key,
                    snapshot=snapshot,
                    tokenizer=custom_tokenizer,
                    refiner=None,
                    config=BuildConfig(
                        include_section_titles=False),
                )

        choose_refiner.assert_called_once()
        apply_refiner.assert_called_once_with(
            unrefined,
            automatic_refiner,
        )
        self.assertIs(write_build.call_args_list[0].args[0], refined)
        self.assertIs(write_build.call_args_list[1].args[0], unrefined)
        self.assertEqual(
            tuple(
                (event.current, event.total)
                for event in events
                if event.phase == "resource"
            ),
            ((0, 1), (1, 1)),
        )

    def test_default_refiner_follows_the_corpus_catalogue_policy(self):
        with (
                patch(
                    "corpus_pipeline.service."
                    "ensure_jieba_traditional_dictionary",
                    return_value=Path("/verified/dict.txt.big"),
                ) as ensure_dictionary,
                patch(
                    "corpus_pipeline.service.JiebaLongSpanRefiner",
                    return_value="refiner",
                ) as refiner_class):
            self.assertIsNone(default_refiner(
                catalogue.DAODEJING,
                Path("/corpora"),
            ))
            selected = default_refiner(
                catalogue.JOURNEY_TO_THE_WEST,
                Path("/corpora"),
            )

        self.assertEqual(selected, "refiner")
        ensure_dictionary.assert_called_once_with(Path("/corpora"))
        refiner_class.assert_called_once_with(
            Path("/verified/dict.txt.big"))

    def test_build_rejects_a_direct_snapshot_that_is_incomplete_for_work(self):
        complete = make_snapshot(
            catalogue.DAODEJING,
            _fetched_pages(),
        )
        first = complete.sections[0]
        incomplete = replace(
            complete,
            sections=(first,),
            canonical_text=first.text,
        )

        with tempfile.TemporaryDirectory() as directory:
            service = CorpusService(directory)
            with self.assertRaisesRegex(
                    CorpusValidationError,
                    "Expected 81 sections"):
                service.build(
                    catalogue.DAODEJING.key,
                    snapshot=incomplete,
                    tokenizer=FakeTokenizer(),
                )

    def test_audit_saved_detects_a_changed_build_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            service = CorpusService(directory)
            fetched = service.fetch(
                catalogue.DAODEJING.key,
                client=_client(),
            )
            result = service.build(
                catalogue.DAODEJING.key,
                snapshot=fetched.snapshot,
                tokenizer=FakeTokenizer(),
                config=BuildConfig(chunk_size=2),
            )
            chunk = next((result.path / "chunks").glob("*.txt"))
            chunk.write_text("篡改\n", encoding="utf-8")

            with self.assertRaisesRegex(
                    ValueError,
                    "artifact hash failed"):
                service.audit_saved(result.path)

    def test_audit_saved_cannot_be_bypassed_by_changing_build_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            service = CorpusService(directory)
            fetched = service.fetch(
                catalogue.DAODEJING.key,
                client=_client(),
            )
            result = service.build(
                catalogue.DAODEJING.key,
                snapshot=fetched.snapshot,
                tokenizer=FakeTokenizer(),
            )
            chunk = next((result.path / "chunks").glob("*.txt"))
            chunk.write_text("篡改\n", encoding="utf-8")
            manifest_path = result.path / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["kind"] = "corpus_snapshot"
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                    ValueError,
                    "artifact hash failed|manifest kind"):
                service.audit_saved(result.path)


class PointerSafetyTests(unittest.TestCase):
    def _write_pointer(self, base, value):
        base.mkdir(parents=True, exist_ok=True)
        (base / "latest_snapshot.json").write_text(
            json.dumps(value),
            encoding="utf-8",
        )

    def test_latest_snapshot_pointer_cannot_escape_corpus_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / catalogue.DAODEJING.key
            outside = root / "outside"
            outside.mkdir()
            self._write_pointer(
                base,
                {
                    "schema_version": CORPUS_SCHEMA_VERSION,
                    "snapshot_id": "outside",
                    "relative_path": "../outside",
                },
            )

            with self.assertRaisesRegex(
                    ValueError,
                    "escapes corpus storage"):
                CorpusService(root).load_latest_snapshot(
                    catalogue.DAODEJING.key
                )

    def test_latest_snapshot_pointer_cannot_escape_through_a_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / catalogue.DAODEJING.key
            outside = root / "outside"
            outside.mkdir()
            link = base / "snapshots" / "linked"
            link.parent.mkdir(parents=True)
            link.symlink_to(outside, target_is_directory=True)
            self._write_pointer(
                base,
                {
                    "schema_version": CORPUS_SCHEMA_VERSION,
                    "snapshot_id": "outside",
                    "relative_path": "snapshots/linked",
                },
            )

            with self.assertRaisesRegex(
                    ValueError,
                    "escapes corpus storage"):
                CorpusService(root).load_latest_snapshot(
                    catalogue.DAODEJING.key
                )

    def test_pointer_identity_must_match_the_selected_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = CorpusService(root)
            fetched = service.fetch(
                catalogue.DAODEJING.key,
                client=_client(),
            )
            pointer_path = (
                root
                / catalogue.DAODEJING.key
                / "latest_snapshot.json"
            )
            pointer = json.loads(
                pointer_path.read_text(encoding="utf-8")
            )
            pointer["snapshot_id"] = "0" * 24
            pointer_path.write_text(
                json.dumps(pointer),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                    ValueError,
                    "identity.*agree|snapshot ID"):
                service.load_latest_snapshot(
                    catalogue.DAODEJING.key
                )

            self.assertTrue(fetched.path.is_dir())

    def test_cached_fetch_rejects_a_relabelled_snapshot_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = CorpusService(root)
            fetched = service.fetch(
                catalogue.DAODEJING.key,
                client=_client(),
            )
            false_identifier = "f" * 24
            false_path = fetched.path.with_name(false_identifier)
            fetched.path.rename(false_path)
            pointer_path = (
                root
                / catalogue.DAODEJING.key
                / "latest_snapshot.json"
            )
            pointer = json.loads(
                pointer_path.read_text(encoding="utf-8"))
            pointer["snapshot_id"] = false_identifier
            pointer["relative_path"] = (
                Path("snapshots") / false_identifier
            ).as_posix()
            pointer_path.write_text(
                json.dumps(pointer),
                encoding="utf-8")
            client = NeverCalledClient()

            with self.assertRaisesRegex(
                    ValueError,
                    "path and content identity"):
                service.fetch(
                    catalogue.DAODEJING.key,
                    client=client)
            self.assertEqual(client.calls, 0)


class CliTests(unittest.TestCase):
    def test_list_is_local_and_reports_both_catalogued_works(self):
        output = io.StringIO()
        with patch("corpus_cli.CorpusService") as service_class:
            with redirect_stdout(output):
                exit_code = corpus_cli.main(["list"])

        self.assertEqual(exit_code, 0)
        service_class.assert_called_once()
        service_class.return_value.fetch.assert_not_called()
        service_class.return_value.build.assert_not_called()
        rendered = output.getvalue()
        self.assertIn("daodejing_wang_bi\t道德經\t王弼本", rendered)
        self.assertIn(
            "daodejing_mawangdui\t道德經\t馬王堆帛書校勘版",
            rendered)
        self.assertIn("journey_to_the_west\t西遊記", rendered)
        self.assertIn("81 sections", rendered)
        self.assertIn("100 sections", rendered)

    def test_help_documents_the_offline_boundary(self):
        output = io.StringIO()
        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                corpus_cli.main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        rendered = output.getvalue()
        self.assertIn("never call OpenAI or Anki", rendered)
        self.assertIn("{list,fetch,build,prepare,audit}", rendered)


if __name__ == "__main__":
    unittest.main()
