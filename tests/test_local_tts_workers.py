from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
WORKERS = REPOSITORY / "scripts" / "tts_workers"
sys.path.insert(0, str(WORKERS))

import _autoanki_tts_worker_common as worker_common
import kokoro_82m_zh_worker
import melotts_jp_worker

INSTALLER_SPEC = importlib.util.spec_from_file_location(
    "autoanki_local_tts_installer",
    REPOSITORY / "scripts" / "install_local_tts.py")
assert INSTALLER_SPEC is not None and INSTALLER_SPEC.loader is not None
installer = importlib.util.module_from_spec(INSTALLER_SPEC)
INSTALLER_SPEC.loader.exec_module(installer)


class LocalTTSWorkerTests(unittest.TestCase):
    def test_melo_source_fingerprint_matches_installer_and_detects_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "melo"
            (source / "text").mkdir(parents=True)
            (source / "api.py").write_text(
                "SOURCE = 'pinned'\n",
                encoding="utf-8")
            (source / "text" / "symbols.txt").write_text(
                "a\nb\n",
                encoding="utf-8")
            initial = installer._source_tree_fingerprint(source)
            self.assertEqual(
                initial,
                melotts_jp_worker._source_tree_fingerprint(source))

            bytecode = source / "__pycache__"
            bytecode.mkdir()
            (bytecode / "api.cpython-310.pyc").write_bytes(b"generated")
            self.assertEqual(
                installer._source_tree_fingerprint(source),
                initial)

            (source / "api.py").write_text(
                "SOURCE = 'modified'\n",
                encoding="utf-8")
            self.assertNotEqual(
                installer._source_tree_fingerprint(source),
                initial)

    def test_melo_accepts_upstream_hparams_speaker_map(self):
        class HParamsLike:
            def __getitem__(self, key):
                if key != "JP":
                    raise KeyError(key)
                return 0

        model = mock.Mock()
        model.hps.data.spk2id = HParamsLike()
        self.assertEqual(melotts_jp_worker._speaker_id(model), 0)

    def test_kokoro_uses_selected_local_voice_and_joins_all_chunks(self):
        class FakeTensor:
            def __init__(self, values):
                self.values = list(values)

            def numel(self):
                return len(self.values)

            def detach(self):
                return self

            def cpu(self):
                return self

            def reshape(self, *_shape):
                return self

        fake_torch = SimpleNamespace(
            Tensor=FakeTensor,
            float32=object(),
            zeros=lambda count, dtype: FakeTensor([0.0] * count),
            cat=lambda tensors: FakeTensor(
                value
                for tensor in tensors
                for value in tensor.values),
            cuda=SimpleNamespace(synchronize=lambda: None),
        )
        pipeline = mock.Mock(return_value=[
            SimpleNamespace(audio=FakeTensor([0.1, 0.2, 0.3])),
            SimpleNamespace(audio=FakeTensor([0.4, 0.5])),
        ])
        voice_path = Path("/installed/voices/zm_010.pt")
        data = {
            "text": "第一句。第二句。",
            "speed": 1.0,
            "seed": 0,
            "sample_rate_hz": 24000,
            "output_path": Path("/staging/output.wav"),
        }
        with (
                mock.patch.object(
                    kokoro_82m_zh_worker,
                    "installed_path",
                    return_value=voice_path),
                mock.patch.object(
                    kokoro_82m_zh_worker,
                    "seed_synthesis"),
                mock.patch.object(
                    kokoro_82m_zh_worker,
                    "write_tensor_wav") as write_wav,
                mock.patch.dict(sys.modules, {"torch": fake_torch})):
            result = kokoro_82m_zh_worker._synthesize_item(
                pipeline,
                {},
                data)

        pipeline.assert_called_once_with(
            data["text"],
            voice=str(voice_path),
            speed=1.0)
        audio, source_rate, target_rate, output_path = (
            write_wav.call_args.args)
        self.assertEqual(audio.numel(), 3 + 1920 + 2)
        self.assertEqual(source_rate, 24000)
        self.assertEqual(target_rate, 24000)
        self.assertEqual(output_path, data["output_path"])
        self.assertEqual(result["artifact"]["path"], str(data["output_path"]))

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

    def test_melo_worker_refresh_records_verified_source_tree_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources" / "MeloTTS" / "revision"
            package = source / "melo"
            package.mkdir(parents=True)
            (package / "api.py").write_text(
                "PINNED = True\n",
                encoding="utf-8")
            tree_revision = installer._source_tree_fingerprint(package)
            runtime = root / "runtimes" / worker_common.MELO_BACKEND
            runtime.mkdir(parents=True)
            manifest_path = runtime / "installation.json"
            manifest_path.write_text(json.dumps({
                "schema_version": 1,
                "backend": worker_common.MELO_BACKEND,
                "state": "ready",
                "paths": {
                    "source": str(source.relative_to(root)),
                },
                "source": {
                    "revision": installer.MELO_SOURCE_REVISION,
                },
                "model": {
                    "id": worker_common.MELO_MODEL_ID,
                    "revision": (
                        "model:model-revision;"
                        "runtime:old-source;"
                        "environment:old-environment;"
                        "worker:old-worker"),
                },
                "runtime": {
                    "torch_requested": "2.11.0",
                    "environment_sha256": "old-environment",
                    "worker_sha256": "old-worker",
                },
            }), encoding="utf-8")
            with (
                    mock.patch.object(
                        installer,
                        "_assert_exact_cuda_torch"),
                    mock.patch.object(
                        installer,
                        "_environment_fingerprint",
                        return_value="e" * 64),
                    mock.patch.object(
                        installer,
                        "_copy_worker"),
                    mock.patch.object(
                        installer,
                        "_worker_fingerprint",
                        return_value="f" * 64)):
                installer._refresh_existing_worker(
                    root,
                    worker_common.MELO_BACKEND)
            refreshed = json.loads(
                manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            refreshed["source"]["tree_sha256"],
            tree_revision)
        self.assertIn(
            (
                f"runtime:{installer.MELO_SOURCE_REVISION}-"
                f"{tree_revision}"
            ),
            refreshed["model"]["revision"].split(";"))
        self.assertEqual(refreshed["state"], "testing")

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
            (
                "melotts_jp_worker.py",
                worker_common.MELO_BACKEND,
                worker_common.MELO_MODEL_ID,
            ),
            (
                "kokoro_82m_zh_worker.py",
                worker_common.KOKORO_BACKEND,
                worker_common.KOKORO_MODEL_ID,
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

    def test_installer_maps_selected_backends_to_exact_workers_and_voices(self):
        self.assertEqual(
            installer._BACKEND_MODEL_IDS[worker_common.MELO_BACKEND],
            worker_common.MELO_MODEL_ID)
        self.assertEqual(
            installer._BACKEND_MODEL_IDS[worker_common.KOKORO_BACKEND],
            worker_common.KOKORO_MODEL_ID)
        self.assertEqual(
            installer._BACKEND_WORKERS[worker_common.MELO_BACKEND].name,
            "melotts_jp_worker.py")
        self.assertEqual(
            installer._BACKEND_WORKERS[worker_common.KOKORO_BACKEND].name,
            "kokoro_82m_zh_worker.py")
        self.assertEqual(
            installer._BACKEND_SMOKE_SAMPLES[worker_common.MELO_BACKEND],
            (("japanese", "こんにちは", "JP"),))
        self.assertEqual(
            installer._BACKEND_SMOKE_SAMPLES[worker_common.KOKORO_BACKEND],
            (("chinese", "你好。", "zm_010"),))
        self.assertEqual(
            tuple(
                language
                for language, _text, _voice
                in installer._BACKEND_SMOKE_SAMPLES[
                    worker_common.COSY_BACKEND]),
            ("english", "french"))
        for requirements in (
                installer.MELO_INFERENCE_REQUIREMENTS,
                installer.KOKORO_INFERENCE_REQUIREMENTS):
            self.assertTrue(all(
                "==" in requirement
                for requirement in requirements))

    def test_installer_production_selection_excludes_optional_style_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            stdout = io.StringIO()
            with (
                    mock.patch.object(
                        installer,
                        "_resolve_python",
                        return_value=Path(sys.executable)),
                    redirect_stdout(stdout)):
                result = installer.main([
                    "--backend",
                    "production",
                    "--root",
                    temporary,
                    "--dry-run",
                ])
        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(stdout.getvalue())["backends"],
            [
                worker_common.COSY_BACKEND,
                worker_common.MELO_BACKEND,
                worker_common.KOKORO_BACKEND,
            ])

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
