from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
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
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(WORKERS))

import _autoanki_tts_worker_common as worker_common
import fun_cosyvoice3_worker
import kokoro_82m_zh_worker
import local_gpu_lease
import melotts_jp_worker
import style_bert_vits2_worker

INSTALLER_SPEC = importlib.util.spec_from_file_location(
    "autoanki_local_tts_installer",
    REPOSITORY / "scripts" / "install_local_tts.py")
assert INSTALLER_SPEC is not None and INSTALLER_SPEC.loader is not None
installer = importlib.util.module_from_spec(INSTALLER_SPEC)
INSTALLER_SPEC.loader.exec_module(installer)

EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "autoanki_local_tts_evaluator",
    REPOSITORY / "scripts" / "evaluate_local_tts_candidates.py")
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
evaluator = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(evaluator)


class LocalTTSWorkerTests(unittest.TestCase):
    def test_cuda_status_inventory_holds_the_shared_model_lease(self):
        events = []

        @contextmanager
        def lease(backend, *, purpose="tts-synthesis"):
            events.append(("lease-acquired", backend, purpose))
            try:
                yield
            finally:
                events.append("lease-released")

        def inventory():
            events.append("cuda-inventory")
            return True, "fixture-gpu", None

        with (
                mock.patch.object(
                    worker_common,
                    "gpu_synthesis_lease",
                    lease),
                mock.patch.object(
                    worker_common,
                    "cuda_inventory",
                    inventory)):
            result = worker_common.leased_cuda_inventory(
                worker_common.COSY_BACKEND)

        self.assertEqual(result, (True, "fixture-gpu", None))
        self.assertEqual(events, [
            (
                "lease-acquired",
                worker_common.COSY_BACKEND,
                "tts-status",
            ),
            "cuda-inventory",
            "lease-released",
        ])

    def test_every_worker_status_routes_cuda_probe_through_shared_lease(self):
        cases = (
            (
                style_bert_vits2_worker,
                worker_common.STYLE_BACKEND,
                "_installation_and_voice",
            ),
            (
                fun_cosyvoice3_worker,
                worker_common.COSY_BACKEND,
                "_installation_and_voices",
            ),
            (
                melotts_jp_worker,
                worker_common.MELO_BACKEND,
                "_installation_and_voice",
            ),
            (
                kokoro_82m_zh_worker,
                worker_common.KOKORO_BACKEND,
                "_installation_and_voice",
            ),
        )
        for module, backend, installation_loader in cases:
            with self.subTest(backend=backend):
                probes = []
                with (
                        mock.patch.object(
                            module,
                            installation_loader,
                            side_effect=RuntimeError("not installed")),
                        mock.patch.object(
                            module,
                            "leased_cuda_inventory",
                            side_effect=lambda selected: (
                                probes.append(selected)
                                or (True, "fixture-gpu", None)
                            )),
                        mock.patch.object(
                            module,
                            "cuda_inventory",
                            side_effect=AssertionError(
                                "status bypassed the shared GPU lease"))):
                    response = module._status()

                self.assertEqual(probes, [backend])
                self.assertTrue(response["gpu"]["available"])

    def test_worker_sources_match_smoke_qualified_identities(self):
        expected = {
            worker_common.STYLE_BACKEND:
                "bfd81807184017daa438f1442f7dbe9365c8951a5f146d146218dcebcb7298e6",
            worker_common.COSY_BACKEND:
                "29b1c1ec3600c613ccc8251b1437d4c96f8c20aa9aa1c00a89b755af64fdd17b",
            worker_common.MELO_BACKEND:
                "f68fe663bf6f6556a8155d9996710845c9ead644d848c060f4255fe9f431da6a",
            worker_common.KOKORO_BACKEND:
                "8e00ba49c26d3ced51f768665d15b840d9914f90919562907a6e9730c0756a6e",
        }
        workers = {
            worker_common.STYLE_BACKEND: "style_bert_vits2_worker.py",
            worker_common.COSY_BACKEND: "fun_cosyvoice3_worker.py",
            worker_common.MELO_BACKEND: "melotts_jp_worker.py",
            worker_common.KOKORO_BACKEND: "kokoro_82m_zh_worker.py",
        }
        common = WORKERS / "_autoanki_tts_worker_common.py"
        for backend, worker_name in workers.items():
            with self.subTest(backend=backend):
                digest = hashlib.sha256()
                for installed_name, source in (
                        ("_autoanki_tts_worker_common.py", common),
                        ("autoanki_worker.py", WORKERS / worker_name)):
                    digest.update(installed_name.encode("utf-8"))
                    digest.update(b"\0")
                    digest.update(source.read_bytes())
                    digest.update(b"\0")
                self.assertEqual(digest.hexdigest(), expected[backend])

    def test_installer_cuda_runtime_check_holds_the_shared_model_lease(self):
        events = []
        runtime_python = Path("/fixture/runtime/bin/python")
        environment = {
            "LOCAL_LLM_GPU_LOCK_PATH": "/fixture/shared/gpu.lock",
        }

        @contextmanager
        def lease(purpose, *, lock_path=None):
            events.append(("lease-acquired", purpose, lock_path))
            try:
                yield
            finally:
                events.append("lease-released")

        def run(*_args, **_kwargs):
            events.append("cuda-runtime-check")

        with (
                mock.patch.object(installer, "local_llm_gpu_lease", lease),
                mock.patch.object(installer, "_run", run)):
            installer._assert_exact_cuda_torch(
                runtime_python,
                expected_version="2.11.0",
                environment=environment)

        self.assertEqual(events, [
            (
                "lease-acquired",
                "tts-runtime-check:runtime",
                Path("/fixture/shared/gpu.lock"),
            ),
            "cuda-runtime-check",
            "lease-released",
        ])

    def test_direct_cuda_evaluator_holds_the_shared_model_lease(self):
        events = []

        @contextmanager
        def lease(purpose):
            events.append(("lease-acquired", purpose))
            try:
                yield
            finally:
                events.append("lease-released")

        def run(_arguments):
            events.append("model-run")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                    mock.patch.object(
                        evaluator,
                        "local_llm_gpu_lease",
                        lease),
                    mock.patch.object(evaluator, "_run_qwen", run),
                    mock.patch.object(evaluator, "_reset_cuda_peak")):
                result = evaluator.main([
                    "qwen",
                    "--output",
                    str(root / "output"),
                    "--asset-root",
                    str(root / "assets"),
                ])

        self.assertEqual(result, 0)
        self.assertEqual(events, [
            ("lease-acquired", "tts-evaluation:qwen"),
            "model-run",
            "lease-released",
        ])

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

    def test_worker_refresh_preserves_environment_without_runtime_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = (
                root / "runtimes" / worker_common.COSY_BACKEND)
            runtime.mkdir(parents=True)
            manifest_path = runtime / "installation.json"
            environment_revision = "e" * 64
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
                        f"environment:{environment_revision};"
                        "worker:old-worker"),
                },
                "runtime": {
                    "torch_requested": "2.11.0",
                    "environment_sha256": environment_revision,
                    "worker_sha256": "old-worker",
                },
            }), encoding="utf-8")
            call_order = []

            def copy_worker(*_args, **_kwargs):
                call_order.append("copy-worker")

            def fingerprint_worker(*_args, **_kwargs):
                call_order.append("fingerprint-worker")
                return "f" * 64

            with (
                    mock.patch.object(
                        installer,
                        "_assert_exact_cuda_torch",
                        side_effect=AssertionError(
                            "Static refresh initialized CUDA.")),
                    mock.patch.object(
                        installer,
                        "_environment_fingerprint",
                        side_effect=AssertionError(
                            "Static refresh inspected the runtime.")),
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
            "copy-worker",
            "fingerprint-worker",
        ])
        self.assertEqual(refreshed["state"], "testing")
        self.assertNotIn("verified_at", refreshed)
        self.assertNotIn("smoke_audio_directory", refreshed)
        self.assertEqual(
            refreshed["runtime"]["environment_sha256"],
            environment_revision)
        self.assertEqual(
            refreshed["runtime"]["worker_sha256"],
            "f" * 64)
        self.assertIn(
            f"environment:{environment_revision}",
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
            environment_revision = "e" * 64
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
                        f"environment:{environment_revision};"
                        "worker:old-worker"),
                },
                "runtime": {
                    "torch_requested": "2.11.0",
                    "environment_sha256": environment_revision,
                    "worker_sha256": "old-worker",
                },
            }), encoding="utf-8")
            with (
                    mock.patch.object(
                        installer,
                        "_assert_exact_cuda_torch",
                        side_effect=AssertionError(
                            "Static refresh initialized CUDA.")),
                    mock.patch.object(
                        installer,
                        "_environment_fingerprint",
                        side_effect=AssertionError(
                            "Static refresh inspected the runtime.")),
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
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                **os.environ,
                "AUTOANKI_TTS_HOME": temporary,
            }
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
                        env=environment,
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
                    {"AUTOANKI_TTS_HOME": str(root)},
                    clear=True):
                with worker_common.gpu_synthesis_lease(
                        worker_common.COSY_BACKEND):
                    worker_lock = root / "locks" / "worker-gpu.lock"
                    self.assertTrue(worker_lock.is_file())
                    self.assertEqual(
                        worker_common.local_llm_gpu_lock_path(),
                        local_gpu_lease.local_llm_gpu_lock_path())
                    self.assertFalse(
                        (root / "locks" / "orchestration.lock").exists())

    def test_worker_gpu_lease_honors_the_host_wide_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tts"
            override = Path(temporary) / "host" / "gpu.lock"
            with mock.patch.dict(os.environ, {
                    "AUTOANKI_TTS_HOME": str(root),
                    "LOCAL_LLM_GPU_LOCK_PATH": str(override),
            }, clear=True):
                with worker_common.gpu_synthesis_lease(
                        worker_common.KOKORO_BACKEND):
                    self.assertTrue(override.is_file())
                    metadata = json.loads(
                        override.read_text(encoding="utf-8"))
                    self.assertEqual(metadata["owner"], "autoanki")
                    self.assertEqual(
                        metadata["purpose"],
                        "tts-synthesis:kokoro_82m_zh")
                    self.assertFalse(
                        (root / "locks" / "worker-gpu.lock").exists())

    def test_cosy_worker_lease_has_no_swap_or_host_memory_gate(self):
        import fcntl

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "locks" / "worker-gpu.lock"
            with mock.patch.dict(
                    os.environ,
                    {"AUTOANKI_TTS_HOME": str(root)},
                    clear=True), mock.patch.object(
                        Path,
                        "read_text",
                        side_effect=AssertionError(
                            "The worker lease read host-memory metrics.")):
                with worker_common.gpu_synthesis_lease(
                        worker_common.COSY_BACKEND):
                    self.assertTrue(path.is_file())
            with path.open("r+b") as stream:
                fcntl.flock(
                    stream.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def test_worker_gpu_lease_refuses_a_relative_host_override(self):
        with mock.patch.dict(os.environ, {
                "LOCAL_LLM_GPU_LOCK_PATH": "relative/gpu.lock",
        }, clear=True):
            with self.assertRaisesRegex(
                    worker_common.RequestError,
                    "absolute path"):
                with worker_common.gpu_synthesis_lease(
                        worker_common.COSY_BACKEND):
                    self.fail("Unsafe relative GPU lease was acquired.")

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
