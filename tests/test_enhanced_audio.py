import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import enhanced_audio


VOICE_SHA = hashlib.sha256(b"neutral voice revision").hexdigest()


class FakeWorker:
    def __init__(self, *, gpu_available=True, payload=b"RIFFfake-audio"):
        self.gpu_available = gpu_available
        self.payload = payload
        self.status_calls = []
        self.synthesis_calls = []

    def status(self, route):
        self.status_calls.append(route)
        return enhanced_audio.TTSBackendStatus(
            backend=route.backend,
            model_id=route.model_id,
            model_revision="model-commit-012345",
            runtime_available=True,
            gpu_available=self.gpu_available,
            device_name=("Test CUDA GPU" if self.gpu_available else None),
            voices={route.default_voice: VOICE_SHA},
            message=(
                "Ready for GPU synthesis."
                if self.gpu_available
                else "CUDA is unavailable."))

    def synthesize(self, request, output_path):
        self.synthesis_calls.append(request)
        output_path.write_bytes(self.payload)


class BatchFakeWorker(FakeWorker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_calls = []

    def synthesize_many(self, requests, output_paths):
        self.batch_calls.append(tuple(requests))
        for request, output_path in zip(
                requests,
                output_paths,
                strict=True):
            self.synthesis_calls.append(request)
            output_path.write_bytes(self.payload)


class EnhancedAudioTests(unittest.TestCase):
    def test_text_is_unescaped_stripped_collapsed_and_nfc_normalized(self):
        value = (
            "  Cafe\u0301 <strong>&amp;amp;</strong>"
            "<script>do not speak me</script><br>  道  ")
        self.assertEqual(
            enhanced_audio.normalize_tts_text(value),
            "Café & 道")
        self.assertEqual(
            enhanced_audio.normalize_tts_text("日<strong>本</strong>語"),
            "日本語")

    def test_audio_first_choice_is_stable_and_approximately_half(self):
        identities = tuple(
            ("deck-seed", "word_to_meaning", index)
            for index in range(10_000))
        first_pass = tuple(
            enhanced_audio.deterministic_audio_first(*identity)
            for identity in identities)
        second_pass = tuple(
            enhanced_audio.deterministic_audio_first(*identity)
            for identity in identities)

        self.assertEqual(first_pass, second_pass)
        audio_first_count = sum(first_pass)
        self.assertGreaterEqual(audio_first_count, 4_700)
        self.assertLessEqual(audio_first_count, 5_300)
        self.assertTrue(all(
            len(enhanced_audio.deterministic_presentation_key(*identity))
            == 64
            for identity in identities[:10]))

    def test_routes_requested_languages_and_rejects_unsupported_archaic_ones(
            self):
        self.assertEqual(
            enhanced_audio.resolve_tts_route("ja-JP").backend,
            enhanced_audio.STYLE_BERT_BACKEND)
        for language in (
                "English",
                "French",
                "Chinese",
                "Traditional Chinese",
                "zh-Hant",
                "Classical Chinese (Wang Bi recension)",
                "classical_chinese_warring_states"):
            with self.subTest(language=language):
                self.assertEqual(
                    enhanced_audio.resolve_tts_route(language).backend,
                    enhanced_audio.COSYVOICE_BACKEND)
        for language in (
                "Middle English",
                "Old English",
                "Latin"):
            with self.subTest(language=language):
                with self.assertRaisesRegex(
                        enhanced_audio.UnsupportedTTSLanguageError,
                        "archaic language"):
                    enhanced_audio.resolve_tts_route(language)

    def test_shared_paths_are_not_inside_the_repository(self):
        with patch.dict(os.environ, {}, clear=True):
            paths = enhanced_audio.TTSPaths.shared()
        self.assertEqual(
            paths.root,
            Path.home() / ".local" / "share" / "autoanki" / "tts")
        self.assertEqual(
            paths.huggingface_cache,
            Path.home() / ".cache" / "huggingface")
        self.assertFalse(paths.root.is_relative_to(PROJECT_ROOT))

    def test_tts_home_and_standard_hf_cache_can_be_overridden(self):
        with patch.dict(os.environ, {
                "AUTOANKI_TTS_HOME": "/tmp/shared-autoanki-tts",
                "HF_HOME": "/tmp/shared-hf-cache",
        }, clear=True):
            paths = enhanced_audio.TTSPaths.shared()
        self.assertEqual(
            paths.root,
            Path("/tmp/shared-autoanki-tts"))
        self.assertEqual(
            paths.huggingface_cache,
            Path("/tmp/shared-hf-cache"))

    def test_identical_requests_are_synthesized_once_and_cached(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = enhanced_audio.TTSPaths.shared(
                Path(temporary_directory) / "tts")
            worker = FakeWorker()
            service = enhanced_audio.LocalTTSService(
                paths=paths,
                worker=worker)
            request = enhanced_audio.AudioRequest(
                "<b>日</b>本語",
                "Japanese",
                "word")
            normalized_duplicate = enhanced_audio.AudioRequest(
                "日本語",
                "ja-JP",
                "word")

            plan = service.plan((request, normalized_duplicate))
            self.assertEqual(plan.requested_count, 2)
            self.assertEqual(plan.unique_count, 1)
            self.assertEqual(plan.cache_hit_count, 0)
            self.assertEqual(plan.synthesis_count, 1)
            first, duplicate = service.execute_plan(plan)

            self.assertEqual(first, duplicate)
            self.assertEqual(len(worker.synthesis_calls), 1)
            self.assertEqual(worker.synthesis_calls[0].text, "日本語")
            self.assertEqual(first.path.read_bytes(), worker.payload)
            self.assertTrue(first.manifest_path.is_file())

            second_worker = FakeWorker(payload=b"different")
            second_service = enhanced_audio.LocalTTSService(
                paths=paths,
                worker=second_worker)
            cached_plan = second_service.plan((request,))
            self.assertEqual(cached_plan.cache_hit_count, 1)
            self.assertEqual(cached_plan.synthesis_count, 0)
            cached = second_service.execute_plan(cached_plan)[0]
            self.assertEqual(cached.sha256, first.sha256)
            self.assertEqual(second_worker.synthesis_calls, [])

    def test_distinct_cache_misses_load_each_backend_once_per_batch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = BatchFakeWorker()
            service = enhanced_audio.LocalTTSService(
                paths=enhanced_audio.TTSPaths.shared(
                    Path(temporary_directory) / "tts"),
                worker=worker)

            artifacts = service.synthesize_many((
                enhanced_audio.AudioRequest(
                    "bonjour",
                    "French",
                    "word"),
                enhanced_audio.AudioRequest(
                    "une phrase",
                    "French",
                    "sentence"),
            ))

            self.assertEqual(len(artifacts), 2)
            self.assertEqual(len(worker.batch_calls), 1)
            self.assertEqual(len(worker.batch_calls[0]), 2)

    def test_large_work_is_bounded_and_keeps_request_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = BatchFakeWorker()
            service = enhanced_audio.LocalTTSService(
                paths=enhanced_audio.TTSPaths.shared(
                    Path(temporary_directory) / "tts"),
                worker=worker)
            count = enhanced_audio.MAX_SYNTHESIS_BATCH_ITEMS + 3
            requests = tuple(
                enhanced_audio.AudioRequest(
                    f"word {index}",
                    "English",
                    "word")
                for index in range(count))

            artifacts = service.synthesize_many(requests)

            self.assertEqual(
                [len(batch) for batch in worker.batch_calls],
                [enhanced_audio.MAX_SYNTHESIS_BATCH_ITEMS, 3])
            self.assertEqual(
                [artifact.cache_key for artifact in artifacts],
                [
                    request.cache_key
                    for batch in worker.batch_calls
                    for request in batch
                ])

    def test_cache_identity_covers_every_material_synthesis_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = enhanced_audio.TTSPaths.shared(
                Path(temporary_directory) / "tts")
            worker = FakeWorker()
            service = enhanced_audio.LocalTTSService(
                paths=paths,
                worker=worker)
            base = dict(
                text="bonjour",
                language="French",
                role="word")
            requests = (
                enhanced_audio.AudioRequest(**base),
                enhanced_audio.AudioRequest(**{
                    **base,
                    "role": "sentence",
                }),
                enhanced_audio.AudioRequest(**{
                    **base,
                    "settings": {
                        **enhanced_audio._default_settings(),
                        "speed": 1.1,
                    },
                }),
                enhanced_audio.AudioRequest(**{
                    **base,
                    "output_format": enhanced_audio.OutputFormat(
                        codec="flac",
                        encoder="flac",
                        sample_rate_hz=24000),
                }),
            )
            artifacts = service.synthesize_many(requests)

            self.assertEqual(len({item.cache_key for item in artifacts}), 4)
            for call in worker.synthesis_calls:
                identity = call.identity()
                self.assertEqual(
                    call.cache_key,
                    enhanced_audio._canonical_digest(identity))
                self.assertIn("revision", identity["model"])
                self.assertIn("revision_sha256", identity["voice"])
                self.assertIn("reference_sha256", identity["voice"])
                self.assertIn("settings", identity)
                self.assertIn("encoder", identity["output"])

    def test_reference_audio_content_not_its_path_affects_cache_key(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_reference = root / "first.wav"
            second_reference = root / "second.wav"
            first_reference.write_bytes(b"same voice")
            second_reference.write_bytes(b"same voice")
            worker = FakeWorker()
            service = enhanced_audio.LocalTTSService(
                paths=enhanced_audio.TTSPaths.shared(root / "tts"),
                worker=worker)
            requests = (
                enhanced_audio.AudioRequest(
                    "hello",
                    "English",
                    "word",
                    reference_audio_path=first_reference),
                enhanced_audio.AudioRequest(
                    "hello",
                    "English",
                    "word",
                    reference_audio_path=second_reference),
            )

            first, second = service.synthesize_many(requests)

            self.assertEqual(first.cache_key, second.cache_key)
            self.assertEqual(len(worker.synthesis_calls), 1)

    def test_cpu_is_never_used_as_a_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = FakeWorker(gpu_available=False)
            service = enhanced_audio.LocalTTSService(
                paths=enhanced_audio.TTSPaths.shared(
                    Path(temporary_directory) / "tts"),
                worker=worker)

            with self.assertRaisesRegex(
                    enhanced_audio.TTSRuntimeUnavailableError,
                    "CUDA"):
                service.synthesize(enhanced_audio.AudioRequest(
                    "hello",
                    "English",
                    "word"))

            self.assertEqual(worker.synthesis_calls, [])

    def test_default_missing_runtime_returns_clear_status(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = enhanced_audio.TTSPaths.shared(
                Path(temporary_directory) / "tts")
            worker = enhanced_audio.JSONSubprocessWorker(paths=paths)
            status = worker.status(
                enhanced_audio.resolve_tts_route("Japanese"))
            self.assertFalse(status.ready)
            self.assertFalse(status.runtime_available)
            self.assertIn("missing", status.message.lower())

    def test_json_subprocess_contract_and_gpu_dispatch(self):
        worker_source = r'''
import hashlib
import json
from pathlib import Path
import sys

request = json.loads(sys.stdin.readline())
base = {
    "protocol": request["protocol"],
    "version": request["version"],
    "operation": request["operation"],
    "backend": request.get("backend", "fun_cosyvoice3_0_5b"),
    "ok": True,
}
if request["operation"] == "status":
    base.update({
        "runtime": {"available": True},
        "gpu": {
            "available": True,
            "device": "cuda",
            "name": "Fake CUDA",
        },
        "model": {
            "id": request["model"]["id"],
            "revision": "fake-model-revision",
        },
        "voices": {
            "neutral-english": {
                "revision_sha256": hashlib.sha256(b"voice").hexdigest(),
            },
        },
    })
elif request["operation"] == "synthesize_many":
    artifacts = []
    for item in request["items"]:
        assert item["execution"] == {
            "required_device": "cuda",
            "allow_cpu_fallback": False,
        }
        output = item["output"]
        Path(output["path"]).write_bytes(b"RIFFsubprocess")
        artifacts.append({
            "cache_key": item["cache_key"],
            "path": output["path"],
            "device": "cuda",
            "codec": output["codec"],
            "encoder": output["encoder"],
        })
    base["artifacts"] = artifacts
else:
    assert request["execution"] == {
        "required_device": "cuda",
        "allow_cpu_fallback": False,
    }
    output = request["output"]
    Path(output["path"]).write_bytes(b"RIFFsubprocess")
    base["artifact"] = {
        "path": output["path"],
        "device": "cuda",
        "codec": output["codec"],
        "encoder": output["encoder"],
    }
print(json.dumps(base))
'''
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            worker_path = root / "fake_worker.py"
            worker_path.write_text(worker_source, encoding="utf-8")
            paths = enhanced_audio.TTSPaths.shared(root / "tts")
            command = enhanced_audio.WorkerCommand(
                (sys.executable, str(worker_path)),
                required_paths=(worker_path,))
            worker = enhanced_audio.JSONSubprocessWorker(
                commands={
                    enhanced_audio.COSYVOICE_BACKEND: command,
                },
                paths=paths)
            service = enhanced_audio.LocalTTSService(
                paths=paths,
                worker=worker)

            artifact = service.synthesize(
                enhanced_audio.AudioRequest(
                    "Hello",
                    "English",
                    "sentence"))

            self.assertEqual(artifact.path.read_bytes(), b"RIFFsubprocess")
            manifest = json.loads(
                artifact.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["request"]["model"]["revision"],
                "fake-model-revision")
            self.assertEqual(
                manifest["request"]["output"]["encoder"],
                "pcm_s16le")

    def test_materialize_media_deduplicates_and_refuses_collisions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            service = enhanced_audio.LocalTTSService(
                paths=enhanced_audio.TTSPaths.shared(root / "tts"),
                worker=FakeWorker())
            artifact = service.synthesize(enhanced_audio.AudioRequest(
                "hello",
                "English",
                "word"))
            destination = root / "package"

            manifest = enhanced_audio.materialize_media(
                (artifact, artifact),
                destination)

            self.assertEqual(len(manifest["artifacts"]), 1)
            target = destination / artifact.media_filename
            self.assertEqual(target.read_bytes(), artifact.path.read_bytes())
            target.write_bytes(b"collision")
            with self.assertRaises(FileExistsError):
                enhanced_audio.materialize_media((artifact,), destination)


if __name__ == "__main__":
    unittest.main()
