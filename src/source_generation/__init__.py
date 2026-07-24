"""Resumable, offline-testable generation from processed source corpora.

This package deliberately does not construct an OpenAI client or talk to Anki.
The caller must inject the paid request operation and, later, explicitly import
the resulting deck.  That boundary keeps planning and cost estimation free.
"""

from source_generation.adapters import SourceGenerationBackend
from source_generation.jobs import (
    GenerationJobRunner,
    GenerationJobStore,
    JobEvent,
    JobSnapshot,
    RetryPolicy,
    is_transient_request_error,
)
from source_generation.models import (
    AnkiExclusionSpec,
    ContextMode,
    ContextUnit,
    CostEstimate,
    GenerationChunk,
    GenerationPlan,
    GenerationWord,
    LoadedSource,
    OutputDetail,
    Pricing,
    SourceGenerationConfig,
)
from source_generation.planning import (
    AUD_EXCHANGE_RATE_LABEL,
    DEFAULT_PRICING,
    MODEL_CONTEXT_WINDOW_TOKENS,
    MODEL_MAX_OUTPUT_TOKENS,
    USD_TO_AUD_RATE,
    estimate_plan_cost,
    estimate_chunk_output_tokens,
    estimate_text_tokens,
    list_processed_source_summaries,
    list_processed_sources,
    load_processed_source,
    plan_source_generation,
)
from source_generation.validation import make_pipeline_response_validator
from source_generation.requests import (
    SOURCE_BATCH_INSTRUCTIONS,
    SOURCE_REQUEST_CONTRACT_SCHEMA_VERSION,
    SOURCE_REQUEST_MODEL,
    SOURCE_REQUEST_REASONING,
    build_source_request_contract,
    build_source_prompt,
    load_source_batch_instructions,
    load_source_web_search_instructions,
    normalise_source_request_contract,
    render_chunk_input,
    source_request_contract_digest,
)

__all__ = (
    "AnkiExclusionSpec",
    "ContextMode",
    "ContextUnit",
    "CostEstimate",
    "AUD_EXCHANGE_RATE_LABEL",
    "DEFAULT_PRICING",
    "MODEL_CONTEXT_WINDOW_TOKENS",
    "MODEL_MAX_OUTPUT_TOKENS",
    "USD_TO_AUD_RATE",
    "GenerationChunk",
    "GenerationJobRunner",
    "GenerationJobStore",
    "GenerationPlan",
    "GenerationWord",
    "JobEvent",
    "JobSnapshot",
    "LoadedSource",
    "OutputDetail",
    "Pricing",
    "RetryPolicy",
    "SourceGenerationConfig",
    "SourceGenerationBackend",
    "SOURCE_BATCH_INSTRUCTIONS",
    "SOURCE_REQUEST_CONTRACT_SCHEMA_VERSION",
    "SOURCE_REQUEST_MODEL",
    "SOURCE_REQUEST_REASONING",
    "build_source_request_contract",
    "build_source_prompt",
    "load_source_batch_instructions",
    "load_source_web_search_instructions",
    "normalise_source_request_contract",
    "estimate_plan_cost",
    "estimate_chunk_output_tokens",
    "estimate_text_tokens",
    "is_transient_request_error",
    "list_processed_source_summaries",
    "list_processed_sources",
    "load_processed_source",
    "make_pipeline_response_validator",
    "plan_source_generation",
    "render_chunk_input",
    "source_request_contract_digest",
)
