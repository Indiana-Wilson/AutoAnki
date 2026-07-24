import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import codex_source_retrieval


class NonClosingStringIO(io.StringIO):
    def close(self):
        self.flush()


class FakeCodexProcess:
    def __init__(
            self,
            command,
            *,
            cwd,
            stdin,
            stdout,
            stderr,
            text,
            encoding):
        del stdin, stdout, stderr, text, encoding
        self.command = command
        self.cwd = Path(cwd)
        self.stdin = NonClosingStringIO()
        self.stdout = iter((
            '{"type":"thread.started","thread_id":"test"}\n',
            '{"type":"turn.completed"}\n',
        ))
        retrieved = self.cwd / "retrieved" / "source.txt"
        retrieved.write_text("道可道。", encoding="utf-8")
        (self.cwd / "result.json").write_text(
            json.dumps({
                "title": "Retrieved work",
                "source_urls": ["https://example.invalid/source"],
                "files": ["retrieved/source.txt"],
                "summary": "Retrieved exact text for inspection.",
                "warnings": [],
            }),
            encoding="utf-8")

    def wait(self):
        return 0


class BlockingCodexProcess(FakeCodexProcess):
    def __init__(self, *args, started, release, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = started
        self.release = release

    @property
    def stdout(self):
        def lines():
            self.started.set()
            self.release.wait(timeout=5)
            yield '{"type":"turn.completed"}\n'

        return lines()

    @stdout.setter
    def stdout(self, _value):
        pass


class CodexSourceRetrievalTests(unittest.TestCase):
    def test_creating_a_job_never_launches_codex(self):
        with tempfile.TemporaryDirectory() as directory:
            path = codex_source_retrieval.create_retrieval_job(
                codex_source_retrieval.CodexRetrievalRequest(
                    description="Find a public-domain original text.",
                    source_title="Test Work",
                    language_key="classical_chinese_ming"),
                output_root=directory)
            manifest = json.loads(
                (path / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["status"], "pending")
        self.assertTrue((path / "request.txt").name, "request.txt")

    def test_explicit_run_is_sandboxed_logged_and_validated(self):
        events = []
        processes = []

        def popen(*args, **kwargs):
            process = FakeCodexProcess(*args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            path = codex_source_retrieval.create_retrieval_job(
                {
                    "description": "Find the requested original text.",
                    "source_title": "Test Work",
                    "language_key": "classical_chinese_ming",
                },
                output_root=directory)
            result = codex_source_retrieval.run_retrieval_job(
                path,
                codex_executable="/usr/bin/codex",
                event_callback=events.append,
                popen=popen)
            manifest = json.loads(
                (path / "manifest.json").read_text(encoding="utf-8"))
            reused = codex_source_retrieval.run_retrieval_job(
                path,
                codex_executable="/usr/bin/codex",
                popen=lambda *_args, **_kwargs: self.fail(
                    "A completed retrieval must not launch Codex again."))

            self.assertEqual(
                result.files,
                (path.resolve() / "retrieved" / "source.txt",))
            self.assertEqual(reused, result)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(events), 2)
            self.assertIn(
                "--sandbox",
                processes[0].command)
            self.assertIn(
                "workspace-write",
                processes[0].command)
            self.assertIn(
                "--search",
                processes[0].command)
            self.assertIn(
                "--ignore-user-config",
                processes[0].command)
            self.assertEqual(
                processes[0].stdin.getvalue(),
                (path / "request.txt").read_text(encoding="utf-8"))

    def test_result_paths_cannot_escape_the_retrieval_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            job.mkdir()
            (job / "result.json").write_text(
                json.dumps({
                    "title": "Bad",
                    "source_urls": [],
                    "files": ["../secret.txt"],
                    "summary": "",
                    "warnings": [],
                }),
                encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "escapes"):
                codex_source_retrieval._validated_result(job)

    def test_concurrent_run_is_rejected_without_launching_second_codex(self):
        started = threading.Event()
        release = threading.Event()
        first_result = []
        first_error = []

        def blocking_popen(*args, **kwargs):
            return BlockingCodexProcess(
                *args,
                started=started,
                release=release,
                **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            path = codex_source_retrieval.create_retrieval_job(
                {
                    "description": "Find the requested original text.",
                    "source_title": "Test Work",
                    "language_key": "classical_chinese_ming",
                },
                output_root=directory)

            def run_first():
                try:
                    first_result.append(
                        codex_source_retrieval.run_retrieval_job(
                            path,
                            codex_executable="/usr/bin/codex",
                            popen=blocking_popen))
                except Exception as error:
                    first_error.append(error)

            worker = threading.Thread(target=run_first)
            worker.start()
            self.assertTrue(started.wait(timeout=5))
            try:
                with self.assertRaisesRegex(
                        codex_source_retrieval
                        .CodexRetrievalAlreadyRunningError,
                        "already running"):
                    codex_source_retrieval.run_retrieval_job(
                        path,
                        codex_executable="/usr/bin/codex",
                        popen=lambda *_args, **_kwargs: self.fail(
                            "A second Codex process must not launch."))
                with self.assertRaisesRegex(
                        codex_source_retrieval
                        .CodexRetrievalAlreadyRunningError,
                        "already running"):
                    codex_source_retrieval.create_retrieval_job(
                        {
                            "description": (
                                "Find the requested original text."),
                            "source_title": "Test Work",
                            "language_key": "classical_chinese_ming",
                        },
                        output_root=directory)
            finally:
                release.set()
                worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            self.assertEqual(first_error, [])
            self.assertEqual(len(first_result), 1)

    def test_job_lease_is_released_after_a_run_error(self):
        class FailedCodexProcess(FakeCodexProcess):
            def wait(self):
                return 9

        with tempfile.TemporaryDirectory() as directory:
            path = codex_source_retrieval.create_retrieval_job(
                {
                    "description": "Find the requested original text.",
                    "source_title": "Test Work",
                    "language_key": "classical_chinese_ming",
                },
                output_root=directory)
            with self.assertRaisesRegex(RuntimeError, "status 9"):
                codex_source_retrieval.run_retrieval_job(
                    path,
                    codex_executable="/usr/bin/codex",
                    popen=FailedCodexProcess)

            recovered = codex_source_retrieval.run_retrieval_job(
                path,
                codex_executable="/usr/bin/codex",
                popen=FakeCodexProcess)

            self.assertEqual(recovered.title, "Retrieved work")


if __name__ == "__main__":
    unittest.main()
