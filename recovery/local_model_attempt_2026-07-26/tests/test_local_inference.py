# Archived with the abandoned local-model integration.
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import local_inference
from source_generation.model_catalog import (
    GEMMA4_12B_SOURCE_MODEL,
    OPENAI_SOURCE_MODEL,
    QWEN3_14B_SOURCE_MODEL,
)


class JsonResponse:
    def __init__(self, value):
        self.payload = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, _error_type, _error, _traceback):
        return False

    def read(self):
        return self.payload


def exact_qwen_tag(*, digest=None, quantization="Q4_K_M"):
    return {
        "name": "qwen3:14b",
        "digest": digest or ("a" * 64),
        "size": 9_321_000_000,
        "modified_at": "2026-07-26T10:00:00Z",
        "details": {
            "family": "qwen3",
            "parameter_size": "14.8B",
            "quantization_level": quantization,
        },
    }


def local_request_options(*, reasoning="low"):
    return {
        "model": "qwen3:14b",
        "input": "Return one strict object for 道.",
        "reasoning": {"effort": reasoning},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "fixture",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "translation": {"type": "string"},
                    },
                    "required": ["translation"],
                    "additionalProperties": False,
                },
            },
        },
        "max_output_tokens": 321,
        "extra_body": {
            "options": {
                "num_ctx": 8_192,
                "temperature": 0.3,
                "top_p": 0.9,
                "seed": 17,
            },
            "keep_alive": "5m",
        },
    }


def qwen_gpu_runtime(
        *,
        size=10_000_000_000,
        size_vram=6_000_000_000,
        context_length=8_192,
        digest=None):
    return {
        "models": [{
            "name": "qwen3:14b",
            "model": "qwen3:14b",
            "digest": digest or ("a" * 64),
            "size": size,
            "size_vram": size_vram,
            "context_length": context_length,
        }],
    }


class LocalInferenceDiscoveryTests(unittest.TestCase):
    def test_snapshot_freezes_exact_tag_digest_q4_runtime_and_policy(self):
        tags = {
            "models": [
                exact_qwen_tag(),
                {
                    "name": "unrelated:latest",
                    "digest": "b" * 64,
                },
            ],
        }
        version = {"version": "0.9.6"}
        with (
                patch.dict(
                    os.environ,
                    {
                        "AUTOANKI_OLLAMA_BASE_URL":
                            "http://127.0.0.1:11434/",
                    },
                    clear=False),
                patch.object(
                    local_inference,
                    "urlopen",
                    side_effect=(
                        JsonResponse(tags),
                        JsonResponse(version),
                    )) as mocked_urlopen):
            snapshot = local_inference.ollama_model_snapshot(
                QWEN3_14B_SOURCE_MODEL)

        self.assertEqual(snapshot, {
            "schema_version": 1,
            "provider": "ollama",
            "model_id": QWEN3_14B_SOURCE_MODEL,
            "provider_model": "qwen3:14b",
            "digest": "a" * 64,
            "size": 9_321_000_000,
            "modified_at": "2026-07-26T10:00:00Z",
            "family": "qwen3",
            "parameter_size": "14.8B",
            "quantization_level": "Q4_K_M",
            "ollama_version": "0.9.6",
            "base_url": "http://127.0.0.1:11434",
            "inference_policy": {
                "schema_version": 4,
                "num_ctx": 8_192,
                "temperature": 0.2,
                "automatic_repair_temperature_schedule": [
                    0.2,
                    0.25,
                    0.3,
                    0.35,
                    0.4,
                ],
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.05,
                "keep_alive": "10m",
                "seed_policy":
                    "sha256(job_id,chunk_id,attempt,model_id)",
                "gpu_offload_guard": "api_ps_positive_size_vram",
            },
        })
        self.assertEqual(mocked_urlopen.call_count, 2)
        tag_request = mocked_urlopen.call_args_list[0].args[0]
        version_request = mocked_urlopen.call_args_list[1].args[0]
        self.assertEqual(
            tag_request.full_url,
            "http://127.0.0.1:11434/api/tags")
        self.assertEqual(tag_request.get_method(), "GET")
        self.assertEqual(
            version_request.full_url,
            "http://127.0.0.1:11434/api/version")

    def test_remote_model_needs_no_ollama_snapshot(self):
        with patch.object(local_inference, "urlopen") as mocked_urlopen:
            snapshot = local_inference.ollama_model_snapshot(
                OPENAI_SOURCE_MODEL)

        self.assertIsNone(snapshot)
        mocked_urlopen.assert_not_called()

    def test_missing_or_duplicate_exact_tag_is_rejected_before_version_read(
            self):
        for models in (
                (),
                ({"name": "qwen3:8b", "digest": "a" * 64},),
                ({
                    **exact_qwen_tag(),
                    "model": "qwen3:8b",
                },),
                ({
                    **exact_qwen_tag(),
                    "model": [],
                },),
                (exact_qwen_tag(), exact_qwen_tag()),
                ("not a model object",),
        ):
            with (
                    self.subTest(models=models),
                    patch.object(
                        local_inference,
                        "urlopen",
                        return_value=JsonResponse({"models": models}))
                    as mocked_urlopen,
                    self.assertRaisesRegex(
                        local_inference.LocalInferenceUnavailableError,
                        "not installed")):
                local_inference.ollama_model_snapshot(
                    QWEN3_14B_SOURCE_MODEL)
            self.assertEqual(mocked_urlopen.call_count, 1)

    def test_non_q4_k_m_tag_is_rejected_before_version_read(self):
        with (
                patch.object(
                    local_inference,
                    "urlopen",
                    return_value=JsonResponse({
                        "models": [
                            exact_qwen_tag(quantization="Q3_K_M"),
                        ],
                    })) as mocked_urlopen,
                self.assertRaisesRegex(
                    local_inference.LocalInferenceUnavailableError,
                    "not the required Q4_K_M")):
            local_inference.ollama_model_snapshot(
                QWEN3_14B_SOURCE_MODEL)

        self.assertEqual(mocked_urlopen.call_count, 1)

    def test_malformed_digest_is_rejected_before_version_read(self):
        for digest in (
                None,
                "",
                "a" * 63,
                "A" * 64,
                "z" * 64,
                123,
        ):
            model = exact_qwen_tag(digest="a" * 64)
            model["digest"] = digest
            with (
                    self.subTest(digest=digest),
                    patch.object(
                        local_inference,
                        "urlopen",
                        return_value=JsonResponse({"models": [model]}))
                    as mocked_urlopen,
                    self.assertRaisesRegex(
                        local_inference.LocalInferenceUnavailableError,
                        "invalid model digest")):
                local_inference.ollama_model_snapshot(
                    QWEN3_14B_SOURCE_MODEL)
            self.assertEqual(mocked_urlopen.call_count, 1)

    def test_runtime_verifier_accepts_exact_snapshot_and_rejects_any_change(
            self):
        current = {
            "model_id": QWEN3_14B_SOURCE_MODEL,
            "digest": "a" * 64,
            "quantization_level": "Q4_K_M",
        }
        with patch.object(
                local_inference,
                "ollama_model_snapshot",
                return_value=current) as snapshot:
            self.assertIs(
                local_inference.ensure_matching_ollama_model(
                    QWEN3_14B_SOURCE_MODEL,
                    current),
                current)
            self.assertIs(
                local_inference.ensure_matching_ollama_model(
                    QWEN3_14B_SOURCE_MODEL),
                current)
            with self.assertRaisesRegex(
                    local_inference.LocalInferenceUnavailableError,
                    "changed after"):
                local_inference.ensure_matching_ollama_model(
                    QWEN3_14B_SOURCE_MODEL,
                    {
                        **current,
                        "digest": "b" * 64,
                    })

        self.assertEqual(snapshot.call_count, 3)


class LocalInferenceRequestTests(unittest.TestCase):
    def test_chat_serializes_native_schema_thinking_options_and_usage(self):
        native_response = {
            "model": "qwen3:14b",
            "created_at": "2026-07-26T10:00:00Z",
            "message": {
                "role": "assistant",
                "content": '{"translation":"the Way"}',
                "thinking": "internal reasoning",
            },
            "done": True,
            "done_reason": "length",
            "total_duration": 8_000,
            "load_duration": 1_000,
            "prompt_eval_count": 11,
            "prompt_eval_duration": 2_000,
            "eval_count": 7,
            "eval_duration": 5_000,
        }
        with patch.object(
                local_inference,
                "urlopen",
                side_effect=(
                    JsonResponse(native_response),
                    JsonResponse(qwen_gpu_runtime()),
                )) as mocked_urlopen:
            result = (
                local_inference.create_ollama_client()
                .responses.create(**local_request_options()))

        self.assertEqual(mocked_urlopen.call_count, 2)
        request = mocked_urlopen.call_args_list[0].args[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:11434/api/chat")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            mocked_urlopen.call_args_list[0].kwargs["timeout"],
            1_800)
        body = json.loads(request.data.decode("utf-8"))
        expected_schema = local_request_options()["text"]["format"]["schema"]
        self.assertEqual(body["model"], "qwen3:14b")
        self.assertEqual(body["messages"], [{
            "role": "user",
            "content": "Return one strict object for 道.",
        }])
        self.assertEqual(body["format"], expected_schema)
        self.assertFalse(body["stream"])
        self.assertTrue(body["think"])
        self.assertEqual(body["keep_alive"], "5m")
        self.assertEqual(body["options"], {
            "num_ctx": 8_192,
            "temperature": 0.3,
            "top_p": 0.9,
            "seed": 17,
            "num_predict": 321,
        })
        self.assertNotIn("num_gpu", body["options"])
        ps_request = mocked_urlopen.call_args_list[1].args[0]
        self.assertEqual(
            ps_request.full_url,
            "http://127.0.0.1:11434/api/ps")
        self.assertEqual(ps_request.get_method(), "GET")

        self.assertTrue(result["id"].startswith("ollama-"))
        self.assertEqual(result["model"], "qwen3:14b")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["service_tier"], "local")
        self.assertEqual(
            result["output_text"],
            '{"translation":"the Way"}')
        self.assertEqual(result["usage"], {
            "input_tokens": 11,
            "input_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "output_tokens": 7,
            "output_tokens_details": {
                "reasoning_tokens": 0,
            },
            "total_tokens": 18,
        })
        self.assertEqual(
            result["local_runtime"]["thinking"],
            "internal reasoning")
        self.assertEqual(result["local_runtime"]["done_reason"], "length")
        self.assertEqual(result["local_runtime"]["digest"], "a" * 64)
        self.assertEqual(result["local_runtime"]["size"], 10_000_000_000)
        self.assertEqual(result["local_runtime"]["size_vram"], 6_000_000_000)
        self.assertEqual(result["local_runtime"]["context_length"], 8_192)
        self.assertEqual(result["local_runtime"]["gpu_fraction"], 0.6)

    def test_no_reasoning_disables_native_thinking_and_defaults_keep_alive(
            self):
        request_options = local_request_options(reasoning="none")
        request_options["extra_body"].pop("keep_alive")
        with patch.object(
                local_inference,
                "urlopen",
                side_effect=(
                    JsonResponse({
                        "model": "qwen3:14b",
                        "message": {"content": "{}"},
                        "done": True,
                        "done_reason": "stop",
                    }),
                    JsonResponse(qwen_gpu_runtime()),
                )) as mocked_urlopen:
            result = (
                local_inference.create_ollama_client()
                .responses.create(**request_options))

        body = json.loads(
            mocked_urlopen.call_args_list[0].args[0].data.decode("utf-8"))
        self.assertFalse(body["think"])
        self.assertEqual(body["keep_alive"], "10m")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["usage"]["total_tokens"], 0)

    def test_gpu_guard_rejects_missing_exact_model_and_cpu_only_runtime(self):
        chat_response = {
            "model": "qwen3:14b",
            "message": {"content": "{}"},
            "done": True,
            "done_reason": "stop",
        }
        cases = (
            (
                {"models": []},
                "exact running model",
            ),
            (
                {
                    "models": [{
                        **qwen_gpu_runtime()["models"][0],
                        "name": "qwen3:8b",
                    }],
                },
                "exact running model",
            ),
            (
                {
                    "models": [{
                        **qwen_gpu_runtime()["models"][0],
                        "model": "qwen3:8b",
                    }],
                },
                "exact running model",
            ),
            (
                {
                    "models": [{
                        **qwen_gpu_runtime()["models"][0],
                        "model": [],
                    }],
                },
                "exact running model",
            ),
            (
                qwen_gpu_runtime(size_vram=0),
                "no GPU offload",
            ),
            (
                qwen_gpu_runtime(size_vram=None),
                "no GPU offload",
            ),
        )
        for ps_response, message in cases:
            with (
                    self.subTest(ps_response=ps_response),
                    patch.object(
                        local_inference,
                        "urlopen",
                        side_effect=(
                            JsonResponse(chat_response),
                            JsonResponse(ps_response),
                        )) as mocked_urlopen,
                    self.assertRaisesRegex(
                        local_inference.LocalInferenceUnavailableError,
                        message)):
                (
                    local_inference.create_ollama_client()
                    .responses.create(**local_request_options()))
            self.assertEqual(mocked_urlopen.call_count, 2)

    def test_gpu_guard_rejects_incomplete_allocation_metadata(self):
        chat_response = {
            "model": "qwen3:14b",
            "message": {"content": "{}"},
            "done": True,
            "done_reason": "stop",
        }
        for changed in (
                {"size": 0},
                {"size": None},
                {"size_vram": 10_000_000_001},
                {"context_length": 0},
                {"context_length": None},
        ):
            runtime = {
                **qwen_gpu_runtime()["models"][0],
                **changed,
            }
            with (
                    self.subTest(changed=changed),
                    patch.object(
                        local_inference,
                        "urlopen",
                        side_effect=(
                            JsonResponse(chat_response),
                            JsonResponse({"models": [runtime]}),
                        )),
                    self.assertRaisesRegex(
                        local_inference.LocalInferenceUnavailableError,
                        "allocation metadata")):
                (
                    local_inference.create_ollama_client()
                    .responses.create(**local_request_options()))

    def test_gpu_guard_binds_response_model_digest_and_requested_context(
            self):
        base_response = {
            "model": "qwen3:14b",
            "message": {"content": "{}"},
            "done": True,
            "done_reason": "stop",
        }
        cases = (
            (
                {
                    **base_response,
                    "model": "qwen3:8b",
                },
                qwen_gpu_runtime(),
                "different model",
                1,
            ),
            (
                base_response,
                qwen_gpu_runtime(digest="not-a-digest"),
                "running-model digest",
                2,
            ),
            (
                base_response,
                qwen_gpu_runtime(context_length=4_096),
                "different running context",
                2,
            ),
        )
        for chat_response, ps_response, message, call_count in cases:
            with (
                    self.subTest(message=message),
                    patch.object(
                        local_inference,
                        "urlopen",
                        side_effect=(
                            JsonResponse(chat_response),
                            JsonResponse(ps_response),
                        )) as mocked_urlopen,
                    self.assertRaisesRegex(
                        local_inference.LocalInferenceUnavailableError,
                        message)):
                (
                    local_inference.create_ollama_client()
                    .responses.create(**local_request_options()))
            self.assertEqual(mocked_urlopen.call_count, call_count)

    def test_invalid_schema_reasoning_and_output_limit_fail_before_network(
            self):
        cases = (
            (
                {"text": {"format": {"type": "text"}}},
                "strict JSON schema",
            ),
            (
                {"reasoning": {"effort": "medium"}},
                "effort",
            ),
            (
                {"max_output_tokens": True},
                "positive local output limit",
            ),
        )
        for changed, message in cases:
            request_options = {
                **local_request_options(),
                **changed,
            }
            with (
                    self.subTest(changed=changed),
                    patch.object(
                        local_inference,
                        "urlopen") as mocked_urlopen,
                    self.assertRaisesRegex(ValueError, message)):
                (
                    local_inference.create_ollama_client()
                    .responses.create(**request_options))
            mocked_urlopen.assert_not_called()

    def test_http_error_retains_status_and_server_detail(self):
        error = HTTPError(
            "http://127.0.0.1:11434/api/tags",
            429,
            "Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"runner is busy"}'))
        with (
                patch.object(
                    local_inference,
                    "urlopen",
                    side_effect=error),
                self.assertRaisesRegex(
                    local_inference.LocalInferenceRequestError,
                    "runner is busy") as caught):
            local_inference._read_json(
                "http://127.0.0.1:11434/api/tags")

        self.assertEqual(caught.exception.status_code, 429)

    def test_connection_error_is_actionable_and_chat_maps_it_to_connection(
            self):
        with (
                patch.object(
                    local_inference,
                    "urlopen",
                    side_effect=URLError("connection refused")),
                self.assertRaisesRegex(
                    local_inference.LocalInferenceUnavailableError,
                    "systemctl --user start ollama")):
            local_inference._read_json(
                "http://127.0.0.1:11434/api/tags")

        with (
                patch.object(
                    local_inference,
                    "urlopen",
                    side_effect=URLError("connection refused")),
                self.assertRaisesRegex(
                    ConnectionError,
                    "shared Ollama service")):
            (
                local_inference.create_ollama_client()
                .responses.create(**local_request_options()))


class LocalInferencePolicyTests(unittest.TestCase):
    def test_url_configuration_accepts_plain_http_and_https_only(self):
        for value, expected in (
                (
                    "http://127.0.0.1:11434/",
                    "http://127.0.0.1:11434",
                ),
                (
                    "https://ollama.internal:443",
                    "https://ollama.internal:443",
                ),
        ):
            with (
                    self.subTest(value=value),
                    patch.dict(
                        os.environ,
                        {"AUTOANKI_OLLAMA_BASE_URL": value},
                        clear=False)):
                self.assertEqual(
                    local_inference.ollama_base_url(),
                    expected)

        for value in (
                "ftp://127.0.0.1:11434",
                "127.0.0.1:11434",
                "http://user:secret@127.0.0.1:11434",
                "http://127.0.0.1:11434/unexpected/path",
                "http://127.0.0.1:11434?model=qwen",
                "http://127.0.0.1:11434#fragment",
        ):
            with (
                    self.subTest(value=value),
                    patch.dict(
                        os.environ,
                        {"AUTOANKI_OLLAMA_BASE_URL": value},
                        clear=False),
                    self.assertRaisesRegex(ValueError, "plain HTTP")):
                local_inference.ollama_base_url()

    def test_policy_and_automatic_repair_temperature_schedule_are_bounded(
            self):
        expected = [0.2, 0.25, 0.3, 0.35, 0.4]
        for model_id in (
                GEMMA4_12B_SOURCE_MODEL,
                QWEN3_14B_SOURCE_MODEL):
            with self.subTest(model_id=model_id):
                policy = local_inference.local_inference_policy(model_id)
                self.assertEqual(policy["schema_version"], 4)
                self.assertEqual(
                    policy["automatic_repair_temperature_schedule"],
                    expected)
                self.assertEqual([
                    local_inference.local_automatic_repair_temperature(
                        model_id,
                        ordinal)
                    for ordinal in range(1, 6)
                ], expected)

        self.assertIsNone(
            local_inference.local_inference_policy(OPENAI_SOURCE_MODEL))
        with self.assertRaisesRegex(ValueError, "requires a local model"):
            local_inference.local_automatic_repair_temperature(
                OPENAI_SOURCE_MODEL,
                1)
        for ordinal in (True, 0, 6, 1.5, "1"):
            with (
                    self.subTest(ordinal=ordinal),
                    self.assertRaisesRegex(ValueError, "outside")):
                local_inference.local_automatic_repair_temperature(
                    QWEN3_14B_SOURCE_MODEL,
                    ordinal)


if __name__ == "__main__":
    unittest.main()
