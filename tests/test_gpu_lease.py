import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import local_gpu_lease


class LocalGPULeaseTests(unittest.TestCase):
    def test_default_retains_the_existing_autoanki_worker_path(self):
        with patch.dict(os.environ, {}, clear=True):
            path = local_gpu_lease.local_llm_gpu_lock_path(
                tts_root=Path("/shared/autoanki/tts"))

        self.assertEqual(
            path,
            Path("/shared/autoanki/tts/locks/worker-gpu.lock"))

    def test_absolute_host_override_is_shared_and_relative_value_is_refused(
            self):
        with tempfile.TemporaryDirectory() as temporary:
            override = Path(temporary) / "host-gpu.lock"
            with patch.dict(os.environ, {
                    "LOCAL_LLM_GPU_LOCK_PATH": str(override),
                    "AUTOANKI_TTS_HOME": "/ignored/autoanki/tts",
            }, clear=True):
                self.assertEqual(
                    local_gpu_lease.local_llm_gpu_lock_path(),
                    override)

        with patch.dict(os.environ, {
                "LOCAL_LLM_GPU_LOCK_PATH": "relative/gpu.lock",
        }, clear=True):
            with self.assertRaisesRegex(
                    local_gpu_lease.LocalGPULeaseError,
                    "absolute path"):
                local_gpu_lease.local_llm_gpu_lock_path()

    @unittest.skipUnless(os.name == "posix", "POSIX flock test")
    def test_lease_is_exclusive_owner_only_and_released_after_error(self):
        import fcntl

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "locks" / "gpu.lock"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with local_gpu_lease.local_llm_gpu_lease(
                        "fixture-cuda-model",
                        lock_path=path) as acquired_path:
                    self.assertEqual(acquired_path, path)
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(payload["owner"], "autoanki")
                    self.assertEqual(
                        payload["purpose"],
                        "fixture-cuda-model")
                    self.assertEqual(payload["pid"], os.getpid())
                    self.assertNotIn("admission", payload)
                    self.assertEqual(
                        stat.S_IMODE(path.stat().st_mode),
                        0o600)
                    with path.open("r+b") as competing:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(
                                competing.fileno(),
                                fcntl.LOCK_EX | fcntl.LOCK_NB)
                    raise RuntimeError("injected lease failure")

            with path.open("r+b") as after_release:
                fcntl.flock(
                    after_release.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(after_release.fileno(), fcntl.LOCK_UN)

    @unittest.skipUnless(os.name == "posix", "POSIX flock test")
    def test_lease_does_not_require_swap_or_host_memory_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu.lock"
            with patch.object(
                    Path,
                    "read_text",
                    side_effect=AssertionError(
                        "The serialization lease read host-memory metrics.")):
                with local_gpu_lease.local_llm_gpu_lease(
                        "fixture-cuda-model",
                        lock_path=path):
                    self.assertTrue(path.is_file())

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["purpose"], "fixture-cuda-model")
            self.assertNotIn("admission", payload)

    @unittest.skipUnless(os.name == "posix", "POSIX inherited flock test")
    def test_inherited_descriptor_keeps_lease_after_parent_death(self):
        import fcntl

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "gpu.lock"
            release = root / "release-child"
            child_code = """
import pathlib
import sys
import time

release = pathlib.Path(sys.argv[1])
deadline = time.monotonic() + 10.0
while not release.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
"""
            parent_code = """
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, sys.argv[1])
from local_gpu_lease import local_llm_gpu_lease_handle

lock_path = pathlib.Path(sys.argv[2])
with local_llm_gpu_lease_handle(
        "fixture-orphan-parent",
        lock_path=lock_path) as lease:
    child = subprocess.Popen(
        (sys.executable, "-c", sys.argv[4], sys.argv[3]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        pass_fds=(lease.file_descriptor,))
    print(child.pid, flush=True)
    os._exit(0)
"""
            completed = subprocess.run(
                (
                    sys.executable,
                    "-I",
                    "-c",
                    parent_code,
                    str(PROJECT_ROOT / "src"),
                    str(path),
                    str(release),
                    child_code,
                ),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=True)
            child_pid = int(completed.stdout.strip())
            try:
                self.assertGreater(child_pid, 0)
                with path.open("r+b") as competing:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(
                            competing.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                release.touch()

            deadline = time.monotonic() + 3.0
            while True:
                with path.open("r+b") as competing:
                    try:
                        fcntl.flock(
                            competing.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            self.fail(
                                "Inherited GPU lease was not released after "
                                "the worker exited.")
                        time.sleep(0.01)
                        continue
                    fcntl.flock(competing.fileno(), fcntl.LOCK_UN)
                    break

    @unittest.skipUnless(os.name == "posix", "POSIX inherited flock test")
    def test_parent_context_close_does_not_unlock_inherited_child(self):
        import fcntl

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "gpu.lock"
            release = root / "release-child"
            child_code = """
import pathlib
import sys
import time

release = pathlib.Path(sys.argv[1])
deadline = time.monotonic() + 10.0
while not release.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
"""
            with local_gpu_lease.local_llm_gpu_lease_handle(
                    "fixture-graceful-parent",
                    lock_path=path) as lease:
                child = subprocess.Popen(
                    (sys.executable, "-c", child_code, str(release)),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    pass_fds=(lease.file_descriptor,))

            try:
                with path.open("r+b") as competing:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(
                            competing.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                release.touch()
                child.wait(timeout=3)

            with path.open("r+b") as after_child_exit:
                fcntl.flock(
                    after_child_exit.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(after_child_exit.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    unittest.main()
