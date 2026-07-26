"""Stable source-generation model identities and runtime capabilities."""

from dataclasses import dataclass


OPENAI_SOURCE_MODEL = "gpt-5.4-mini"
DEFAULT_SOURCE_MODEL = OPENAI_SOURCE_MODEL


@dataclass(frozen=True)
class SourceModelProfile:
    model_id: str
    provider: str
    provider_model: str
    display_name: str
    local: bool
    supports_economy: bool
    supports_web_search: bool
    prompt_cache_mode: str
    max_context_tokens: int
    max_input_tokens: int
    max_output_tokens: int
    max_automatic_repairs: int
    default_transient_retries: int
    recommended_chunk_size: int
    recommended_concurrency: int


_SOURCE_MODEL_PROFILES = {
    OPENAI_SOURCE_MODEL: SourceModelProfile(
        model_id=OPENAI_SOURCE_MODEL,
        provider="openai",
        provider_model=OPENAI_SOURCE_MODEL,
        display_name="GPT-5.4 mini",
        local=False,
        supports_economy=True,
        supports_web_search=True,
        prompt_cache_mode="implicit",
        max_context_tokens=400_000,
        max_input_tokens=272_000,
        max_output_tokens=128_000,
        max_automatic_repairs=3,
        default_transient_retries=3,
        recommended_chunk_size=30,
        recommended_concurrency=8,
    ),
}

SUPPORTED_SOURCE_MODELS = frozenset(_SOURCE_MODEL_PROFILES)
LOCAL_SOURCE_MODELS = frozenset(
    model_id
    for model_id, profile in _SOURCE_MODEL_PROFILES.items()
    if profile.local)


def source_model_profile(model_id):
    try:
        return _SOURCE_MODEL_PROFILES[model_id]
    except KeyError as error:
        raise ValueError(
            "Unsupported source-generation model "
            f"{model_id!r}.") from error


def is_local_source_model(model_id):
    return source_model_profile(model_id).local


def provider_model_name(model_id):
    return source_model_profile(model_id).provider_model


def source_model_profiles():
    return tuple(_SOURCE_MODEL_PROFILES.values())
