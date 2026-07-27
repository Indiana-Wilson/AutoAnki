from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
WORKERS = REPOSITORY / "scripts" / "tts_workers"
sys.path.insert(0, str(WORKERS))

import _autoanki_tts_worker_common as worker_common

INSTALLER_SPEC = importlib.util.spec_from_file_location(
    "autoanki_local_tts_installer",
    REPOSITORY / "scripts" / "install_local_tts.py")
assert INSTALLER_SPEC is not None and INSTALLER_SPEC.loader is not None
installer = importlib.util.module_from_spec(INSTALLER_SPEC)
INSTALLER_SPEC.loader.exec_module(installer)


class LocalTTSWorkerTests(unittest.TestCase):
    def test_backend_stdout_is_redirected_away_from_protocol(self):
        request = {
            "protocol": worker_common.PROTOCOL,
            "version": worker_common.VERSION,
            "backend": worker_common.COSY_BACKEND,
            "operation": "status",
            "model": {"id": worker_common.COSY_MODEL_ID},
            "execution": {
                "required_device": "cuda",
                "allow_cpu_fallback": False,
            },
        }
        stdout = io.StringIO()
        stderr = io.StringIO()

        def handler(_request, _operation):
            print("third-party progress banner")
            return {"value": "ok"}

        with (
                mock.patch.object(
                    sys, "stdin", io.StringIO(json.dumps(request) + "\n")),
                redirect_stdout(stdout),
                redirect_stderr(stderr)):
            result = worker_common.run_worker(
                backend=worker_common.COSY_BACKEND,
                model_id=worker_common.COSY_MODEL_ID,
                handler=handler)
        self.assertEqual(result, 0)
        lines = stdout.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["value"], "ok")
        self.assertIn("third-party progress banner", stderr.getvalue())

    def test_cosy_inference_dependencies_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            runtime = root / "runtime"
            source.mkdir()
            runtime.mkdir()
            (source / "requirements.txt").write_text(
                "\n".join((
                    "lightning==2.2.4",
                    "matplotlib==3.7.5",
                    "openai-whisper==20231117",
                    "onnxruntime-gpu==1.18.0",
                    "torch==2.3.1",
                )) + "\n",
                encoding="utf-8")
            filtered = installer._filtered_cosy_requirements(
                source, runtime).read_text(encoding="utf-8")
        self.assertIn("lightning==2.2.4", filtered)
        self.assertIn("matplotlib==3.7.5", filtered)
        self.assertNotIn("openai-whisper", filtered)
        self.assertNotIn("onnxruntime-gpu", filtered)
        self.assertNotIn("torch==", filtered)

    def test_install_phases_are_mutually_exclusive(self):
        installer = REPOSITORY / "scripts" / "install_local_tts.py"
        completed = subprocess.run(
            (
                sys.executable,
                installer,
                "--backend",
                "all",
                "--defer-smoke",
                "--verify-existing",
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("not allowed", completed.stderr)

    def test_worker_refresh_validates_environment_before_mutating_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = (
                root / "runtimes" / worker_common.COSY_BACKEND)
            runtime.mkdir(parents=True)
            manifest_path = runtime / "installation.json"
            manifest_path.write_text(json.dumps({
                "schema_version": 1,
                "backend": worker_common.COSY_BACKEND,
                "state": "ready",
                "verified_at": "old-verification",
                "smoke_audio_directory": "old-smoke",
                "model": {
                    "id": worker_common.COSY_MODEL_ID,
                    "revision": (
                        "model:model-revision;"
                        "environment:old-environment;"
                        "worker:old-worker"),
                },
                "runtime": {
                    "torch_requested": "2.11.0",
                    "environment_sha256": "old-environment",
                    "worker_sha256": "old-worker",
                },
            }), encoding="utf-8")
            call_order = []

            def assert_cuda(*_args, **_kwargs):
                call_order.append("validate-cuda")

            def fingerprint_environment(*_args, **_kwargs):
                call_order.append("fingerprint-environment")
                return "e" * 64

            def copy_worker(*_args, **_kwargs):
                call_order.append("copy-worker")

            def fingerprint_worker(*_args, **_kwargs):
                call_order.append("fingerprint-worker")
                return "f" * 64

            with (
                    mock.patch.object(
                        installer,
                        "_assert_exact_cuda_torch",
                        side_effect=assert_cuda),
                    mock.patch.object(
                        installer,
                        "_environment_fingerprint",
                        side_effect=fingerprint_environment),
                    mock.patch.object(
                        installer,
                        "_copy_worker",
                        side_effect=copy_worker),
                    mock.patch.object(
                        installer,
                        "_worker_fingerprint",
                        side_effect=fingerprint_worker)):
                result = installer._refresh_existing_worker(
                    root,
                    worker_common.COSY_BACKEND)

            refreshed = json.loads(
                manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(result, runtime)
        self.assertEqual(call_order, [
            "validate-cuda",
            "fingerprint-environment",
            "copy-worker",
            "fingerprint-worker",
        ])
        self.assertEqual(refreshed["state"], "testing")
        self.assertNotIn("verified_at", refreshed)
        self.assertNotIn("smoke_audio_directory", refreshed)
        self.assertEqual(
            refreshed["runtime"]["environment_sha256"],
            "e" * 64)
        self.assertEqual(
            refreshed["runtime"]["worker_sha256"],
            "f" * 64)
        self.assertIn(
            f"environment:{'e' * 64}",
            refreshed["model"]["revision"])
        self.assertIn(
            f"worker:{'f' * 64}",
            refreshed["model"]["revision"])

    def test_status_is_machine_readable_without_installation(self):
        cases = (
            (
                "style_bert_vits2_worker.py",
                worker_common.STYLE_BACKEND,
                worker_common.STYLE_MODEL_ID,
            ),
            (
                "fun_cosyvoice3_worker.py",
                worker_common.COSY_BACKEND,
                worker_common.COSY_MODEL_ID,
            ),
        )
        for filename, backend, model_id in cases:
            with self.subTest(backend=backend):
                request = {
                    "protocol": worker_common.PROTOCOL,
                    "version": worker_common.VERSION,
                    "backend": backend,
                    "operation": "status",
                    "model": {"id": model_id},
                    "execution": {
                        "required_device": "cuda",
                        "allow_cpu_fallback": False,
                    },
                }
                completed = subprocess.run(
                    (sys.executable, WORKERS / filename),
                    input=json.dumps(request) + "\n",
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False)
                self.assertEqual(completed.returncode, 0)
                lines = [
                    line for line in completed.stdout.splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(lines), 1)
                response = json.loads(lines[0])
                self.assertTrue(response["ok"])
                self.assertFalse(response["runtime"]["available"])
                self.assertEqual(response["model"]["id"], model_id)

    def test_batch_identity_does_not_require_global_model(self):
        request = {
            "protocol": worker_common.PROTOCOL,
            "version": worker_common.VERSION,
            "backend": worker_common.COSY_BACKEND,
            "operation": "synthesize_many",
            "items": [],
        }
        self.assertEqual(
            worker_common.validate_request_identity(
                request,
                backend=worker_common.COSY_BACKEND,
                model_id=worker_common.COSY_MODEL_ID),
            "synthesize_many")

    def test_worker_gpu_lease_is_distinct_from_parent_orchestration_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.dict(
                    os.environ,
                    {"AUTOANKI_TTS_HOME": str(root)}):
                with worker_common.gpu_synthesis_lease(
                        worker_common.COSY_BACKEND):
                    worker_lock = root / "locks" / "worker-gpu.lock"
                    self.assertTrue(worker_lock.is_file())
                    self.assertFalse(
                        (root / "locks" / "orchestration.lock").exists())

    def test_batch_items_are_prevalidated_and_keep_cache_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            staging.mkdir()
            digest = "a" * 64
            installation = {
                "model": {
                    "id": worker_common.COSY_MODEL_ID,
                    "revision": "immutable",
                },
            }
            item = {
                "cache_key": digest,
                "model": {
                    "id": worker_common.COSY_MODEL_ID,
                    "revision": "immutable",
                },
                "input": {
                    "text": "Bonjour",
                    "language": "french",
                    "role": "word",
                },
                "voice": {
                    "id": "neutral-french",
                    "revision_sha256": digest,
                    "reference_path": None,
                    "reference_sha256": None,
                },
                "settings": {
                    "emotion": "neutral",
                    "speed": 1.0,
                    "pitch": 0.0,
                    "energy": 1.0,
                    "seed": 0,
                },
                "output": {
                    "codec": "wav",
                    "encoder": "pcm_s16le",
                    "sample_rate_hz": 24000,
                    "channels": 1,
                    "path": str((staging / "one.wav").resolve()),
                },
                "execution": {
                    "required_device": "cuda",
                    "allow_cpu_fallback": False,
                },
            }
            with mock.patch.dict(
                    os.environ,
                    {"AUTOANKI_TTS_HOME": str(root)}):
                prepared = worker_common.validate_synthesis_items(
                    {"items": [item]},
                    installation,
                    supported_languages={"french"},
                    voice_revisions={"neutral-french": digest})
            self.assertEqual(prepared[0]["cache_key"], digest)
            artifact = worker_common.success_artifact(
                prepared[0])["artifact"]
            self.assertEqual(artifact["cache_key"], digest)

    def test_batch_rejects_duplicate_cache_keys_before_synthesis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            staging.mkdir()
            digest = "b" * 64
            installation = {
                "model": {
                    "id": worker_common.COSY_MODEL_ID,
                    "revision": "immutable",
                },
            }
            base = {
                "cache_key": digest,
                "model": {
                    "id": worker_common.COSY_MODEL_ID,
                    "revision": "immutable",
                },
                "input": {
                    "text": "Hello",
                    "language": "english",
                    "role": "word",
                },
                "voice": {
                    "id": "neutral-english",
                    "revision_sha256": digest,
                    "reference_path": None,
                    "reference_sha256": None,
                },
                "settings": {
                    "emotion": "neutral",
                    "speed": 1.0,
                    "pitch": 0.0,
                    "energy": 1.0,
                    "seed": 0,
                },
                "execution": {
                    "required_device": "cuda",
                    "allow_cpu_fallback": False,
                },
            }
            items = []
            for index in range(2):
                items.append({
                    **base,
                    "output": {
                        "codec": "wav",
                        "encoder": "pcm_s16le",
                        "sample_rate_hz": 24000,
                        "channels": 1,
                        "path": str(
                            (staging / f"{index}.wav").resolve()),
                    },
                })
            with mock.patch.dict(
                    os.environ,
                    {"AUTOANKI_TTS_HOME": str(root)}):
                with self.assertRaisesRegex(
                        worker_common.RequestError,
                        "distinct cache key"):
                    worker_common.validate_synthesis_items(
                        {"items": items},
                        installation,
                        supported_languages={"english"},
                        voice_revisions={"neutral-english": digest})


if __name__ == "__main__":
    unittest.main()
