"""Tokenizer interfaces and lazy era-specific CKIP Han adapters."""

from contextlib import contextmanager, nullcontext
import gc
from importlib import metadata
import os
import platform
import re
from typing import Protocol
import unicodedata

from corpus_pipeline.models import TokenSpan, TokenizerIdentity
from corpus_pipeline.processing import (
    HISTORICAL_ENGLISH_LANGUAGE_KEYS,
    HISTORICAL_ENGLISH_NORMALIZATION_POLICY,
    historical_english_word_ranges,
    is_han_component,
)
from local_gpu_lease import local_llm_gpu_lease


SHANGGU_MODEL = "ckiplab/bert-base-han-chinese-ws-shanggu"
SHANGGU_REVISION = "7aad76735b4cd91a943fba0fbdc966d923b0980e"
JINDAI_MODEL = "ckiplab/bert-base-han-chinese-ws-jindai"
JINDAI_REVISION = "88fbf9d00da985e2babd1ac6c71ce31f9aa24c61"
CKIP_DECODER_VERSION = "bert-offset-bi-v2"
CKIP_UNITIZER_VERSION = "sentence-isolated-v1"
CKIP_REFINEMENT_VERSION = "isolated-long-han-v1"
CKIP_REFINEMENT_THRESHOLD = 4
CKIP_BATCH_SCHEDULER_VERSION = "length-sorted-v1"
HISTORICAL_ENGLISH_TOKENIZER_VERSION = "unicode-orthographic-words-v2"


_MODEL_SENTENCE_END = re.compile(
    r"[。！？!?]+[」』】）》”’]*[ \t]*(?:\n[ \t]*)*"
    r"|\n[ \t]*\n+(?:[ \t]*\n)*")


class CorpusTokenizerUnavailableError(RuntimeError):
    """An explicitly selected optional tokenizer cannot be loaded."""


def recommended_hardware_batch_size():
    """Return a conservative throughput-oriented batch for available hardware."""
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        return 8
    if torch.cuda.is_available():
        return 32
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return 32
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return 24
    return 128


class Tokenizer(Protocol):
    @property
    def identity(self) -> TokenizerIdentity:
        ...

    def tokenize(self, text: str) -> tuple[TokenSpan, ...]:
        ...


def _installed_version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unavailable"


def split_model_units(text, max_characters=400):
    """Return offset-aligned sentence units with a length-safe fallback.

    CKIP's historical word-segmentation models are highly sensitive to
    unrelated sentences in the same inference window. Keep each sentence
    (or unterminated paragraph) isolated, then split only overlong units at
    the best available natural boundary.
    """
    if max_characters < 32:
        raise ValueError("Model chunks must allow at least 32 characters.")
    preferred = frozenset(
        "。！？!?；;，,、：:\n\r\t ")
    chunks = []

    def append_bounded(start, end):
        while start < end:
            hard_end = min(start + max_characters, end)
            chunk_end = hard_end
            if hard_end < end:
                search_start = start + max_characters // 2
                for candidate in range(
                        hard_end - 1,
                        search_start - 1,
                        -1):
                    if text[candidate] in preferred:
                        chunk_end = candidate + 1
                        break
            chunks.append((start, text[start:chunk_end]))
            start = chunk_end

    start = 0
    for match in _MODEL_SENTENCE_END.finditer(text):
        append_bounded(start, match.end())
        start = match.end()
    append_bounded(start, len(text))
    return tuple(chunks)


class CharacterTokenizer:
    """Deterministic diagnostic tokenizer; not used for real corpus builds."""

    @property
    def identity(self):
        return TokenizerIdentity(
            backend="character-debug",
            backend_version="1",
            model="one-code-point-per-token",
            model_revision="1")

    def tokenize(self, text):
        return tuple(
            TokenSpan(character, index, index + 1, 1.0)
            for index, character in enumerate(text)
            if not character.isspace())


class HistoricalEnglishTokenizer:
    """Dependency-free, offset-exact Middle/Old English word tokenizer."""

    def __init__(self, language_key):
        if language_key not in HISTORICAL_ENGLISH_LANGUAGE_KEYS:
            raise ValueError(
                "Historical English tokenization requires Middle English "
                "or Old English.")
        self.language_key = language_key

    @classmethod
    def for_middle_english(cls):
        return cls("middle_english")

    @classmethod
    def for_old_english(cls):
        return cls("old_english")

    @property
    def identity(self):
        return TokenizerIdentity(
            backend="autoanki-unicode-historical-english",
            backend_version=HISTORICAL_ENGLISH_TOKENIZER_VERSION,
            model=f"{self.language_key}-orthographic-words",
            model_revision="1",
            options=(
                ("editorial_number_policy", "omit-v2"),
                ("normalization", HISTORICAL_ENGLISH_NORMALIZATION_POLICY),
                ("unicode_database", unicodedata.unidata_version),
            ))

    def tokenize(self, text):
        normalized = unicodedata.normalize("NFC", text)
        if normalized != text:
            raise ValueError(
                "Tokenizer input must already use NFC normalization.")
        return tuple(
            TokenSpan(
                surface=text[start:end],
                start_offset=start,
                end_offset=end,
                confidence=1.0)
            for start, end in historical_english_word_ranges(text)
        )


class CkipHanTokenizer:
    """Word segmentation using a pinned CKIP historical-language model."""

    def __init__(
            self,
            model,
            revision,
            *,
            device="auto",
            batch_size=8,
            max_characters=400):
        if (
                isinstance(batch_size, bool)
                or not isinstance(batch_size, int)
                or batch_size < 1):
            raise ValueError(
                "Tokenizer batch size must be a positive integer.")
        if (
                isinstance(max_characters, bool)
                or not isinstance(max_characters, int)
                or not 32 <= max_characters <= 510):
            raise ValueError(
                "Tokenizer model chunks must contain 32 to 510 "
                "characters.")
        if not (
                device in {"auto", "cpu", "cuda", "mps", "xpu"}
                or (
                    isinstance(device, str)
                    and (
                        device.startswith("cuda:")
                        or device.startswith("xpu:"))
                    and device.rsplit(":", 1)[-1].isdigit())):
            raise ValueError(
                "Tokenizer device must be auto, cpu, cuda, cuda:N, "
                "mps, xpu, or xpu:N.")
        self.model_name = model
        self.revision = revision
        self.device = device
        self._resolved_device = None
        self.batch_size = batch_size
        self.max_characters = max_characters
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._execution_session_depth = 0
        self._execution_session_device = None

    @classmethod
    def for_shanggu(cls, **kwargs):
        return cls(SHANGGU_MODEL, SHANGGU_REVISION, **kwargs)

    @classmethod
    def for_jindai(cls, **kwargs):
        return cls(JINDAI_MODEL, JINDAI_REVISION, **kwargs)

    @property
    def identity(self):
        identity_device = self._resolved_device or self.device
        return TokenizerIdentity(
            backend="transformers-ckip-han",
            backend_version=_installed_version("transformers"),
            model=self.model_name,
            model_revision=self.revision,
            options=(
                ("batch_size", str(self.batch_size)),
                ("batch_scheduler_version", CKIP_BATCH_SCHEDULER_VERSION),
                ("decoder_version", CKIP_DECODER_VERSION),
                ("device", identity_device),
                ("max_characters", str(self.max_characters)),
                ("platform_machine", platform.machine()),
                ("platform_system", platform.system()),
                ("python_version", platform.python_version()),
                ("pytorch_version", _installed_version("torch")),
                ("refinement_version", CKIP_REFINEMENT_VERSION),
                ("tokenizers_version", _installed_version("tokenizers")),
                ("unitizer_version", CKIP_UNITIZER_VERSION),
            ))

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from transformers import (
                AutoModelForTokenClassification,
                BertTokenizerFast,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise CorpusTokenizerUnavailableError(
                "Historical Chinese tokenization requires the optional "
                "packages in requirements-corpus.txt. No fallback tokenizer "
                "was used.") from error

        # Retain the runtime before any model or vocabulary operation so an
        # exception during loading can still clear CUDA's allocator cache.
        self._torch = torch
        resolved_device = self._resolve_device(torch)
        tokenizer = None
        model = None
        try:
            vocab_path = hf_hub_download(
                self.model_name,
                "vocab.txt",
                revision=self.revision)
            # The repository's old tokenizer.json triggers a panic in modern
            # versions of the Rust tokenizers library. Building the equivalent
            # standard BERT tokenizer from the pinned vocabulary avoids that
            # incompatibility while retaining exact offset mappings.
            tokenizer = BertTokenizerFast(
                vocab_file=vocab_path,
                do_lower_case=False)
            conversion_setting = os.environ.get(
                "DISABLE_SAFETENSORS_CONVERSION")
            os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "true"
            try:
                model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name,
                    revision=self.revision,
                    use_safetensors=False)
            finally:
                if conversion_setting is None:
                    os.environ.pop(
                        "DISABLE_SAFETENSORS_CONVERSION",
                        None)
                else:
                    os.environ[
                        "DISABLE_SAFETENSORS_CONVERSION"
                    ] = conversion_setting
            if not getattr(tokenizer, "is_fast", False):
                raise CorpusTokenizerUnavailableError(
                    "The CKIP tokenizer did not provide offset mappings.")
            # Publish model references before the CUDA transfer so the outer
            # execution session can clear them even when transfer/eval fails.
            self._tokenizer = tokenizer
            self._model = model
            model.to(resolved_device)
            model.eval()
        except CorpusTokenizerUnavailableError:
            self._model = None
            self._tokenizer = None
            model = None
            tokenizer = None
            raise
        except Exception as error:
            self._model = None
            self._tokenizer = None
            model = None
            tokenizer = None
            raise CorpusTokenizerUnavailableError(
                f"Could not load the pinned tokenizer {self.model_name}. "
                "Check the network/cache and try again manually.") from error

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._resolved_device = resolved_device

    def prepare(self):
        """Load the model and resolve its public reproducibility identity.

        Source-preparation checkpoints call this before deriving their cache
        key. This intentionally avoids relying on the adapter's private
        runtime state when ``device="auto"`` is selected.
        """
        if self._execution_session_depth:
            self._load()
            return self.identity
        with self.execution_session():
            self._load()
            return self.identity

    def _session_device(self):
        if self._resolved_device is not None:
            return self._resolved_device
        if self.device != "auto":
            return self.device
        try:
            import torch
        except (ImportError, ModuleNotFoundError):
            # The normal load path will raise the established actionable
            # optional-dependency error. No CUDA allocation can occur when
            # PyTorch itself is unavailable, so no lease is needed first.
            return None
        # Retain the runtime early so a model-load exception can still empty
        # CUDA's allocator cache before the host lease is released.
        self._torch = torch
        return self._resolve_device(torch)

    def _release_model_resources(self):
        """Drop every model reference before releasing a CUDA lease."""
        torch_runtime = self._torch
        resolved_device = (
            self._resolved_device
            if self._resolved_device is not None
            else self._execution_session_device)
        using_cuda = (
            isinstance(resolved_device, str)
            and resolved_device.startswith("cuda"))
        if using_cuda and torch_runtime is not None:
            try:
                torch_runtime.cuda.synchronize()
            except Exception:
                # Cleanup must still discard model references after a failed
                # CUDA operation. The original inference error remains the
                # useful failure for the caller.
                pass
        self._model = None
        self._tokenizer = None
        self._torch = None
        gc.collect()
        if using_cuda and torch_runtime is not None:
            try:
                torch_runtime.cuda.empty_cache()
            except Exception:
                pass

    @contextmanager
    def execution_session(self):
        """Keep one model and, for CUDA, one host lease for a bounded run."""
        if self._execution_session_depth:
            self._execution_session_depth += 1
            try:
                yield self
            finally:
                self._execution_session_depth -= 1
            return

        # ``device=auto`` itself may import Torch, create a CUDA context, and
        # query the driver. Acquire before resolving it so even that probe is
        # serialized with the other host model workloads.
        requires_probe_lease = self.device == "auto" or (
            isinstance(self.device, str)
            and self.device.startswith("cuda"))
        lease = (
            local_llm_gpu_lease(
                f"ckip-tokenizer:{self.model_name}")
            if requires_probe_lease
            else nullcontext())
        with lease:
            selected_device = self._session_device()
            self._execution_session_device = selected_device
            self._execution_session_depth = 1
            try:
                yield self
            finally:
                try:
                    self._release_model_resources()
                finally:
                    self._execution_session_depth = 0
                    self._execution_session_device = None

    def _resolve_device(self, torch):
        """Choose the fastest requested device without hiding fallbacks."""
        if self.device != "auto":
            requested = self.device
            if requested.startswith("cuda") and not torch.cuda.is_available():
                raise CorpusTokenizerUnavailableError(
                    "CUDA tokenization was selected, but this PyTorch "
                    "installation cannot access an NVIDIA GPU.")
            if requested.startswith("xpu"):
                xpu = getattr(torch, "xpu", None)
                if xpu is None or not xpu.is_available():
                    raise CorpusTokenizerUnavailableError(
                        "Intel XPU tokenization was selected, but this "
                        "PyTorch installation cannot access an Intel GPU.")
            if requested == "mps":
                mps = getattr(torch.backends, "mps", None)
                if mps is None or not mps.is_available():
                    raise CorpusTokenizerUnavailableError(
                        "Apple MPS tokenization was selected, but it is "
                        "unavailable.")
            return requested

        if torch.cuda.is_available():
            return "cuda"
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return "xpu"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _label(self, label_id):
        labels = self._model.config.id2label
        return str(
            labels.get(label_id, labels.get(str(label_id), label_id)))

    def _decode_chunk(
            self,
            source,
            source_offset,
            offsets,
            label_ids,
            scores,
            attention):
        spans = []
        current_start = None
        current_end = None
        current_scores = []

        def finish_current():
            nonlocal current_start, current_end, current_scores
            if current_start is None:
                return
            spans.append(TokenSpan(
                surface=source[current_start:current_end],
                start_offset=source_offset + current_start,
                end_offset=source_offset + current_end,
                confidence=(
                    sum(current_scores) / len(current_scores)
                    if current_scores
                    else None)))
            current_start = None
            current_end = None
            current_scores = []

        for raw_offset, label_id, score, is_attended in zip(
                offsets,
                label_ids,
                scores,
                attention,
                strict=True):
            if not is_attended:
                continue
            start, end = (int(raw_offset[0]), int(raw_offset[1]))
            if start == end:
                continue
            label = self._label(int(label_id)).strip().upper()
            if label not in {"B", "I"}:
                raise CorpusTokenizerUnavailableError(
                    "The pinned CKIP model returned an unsupported word-"
                    f"segmentation label: {label!r}. Expected only B or I.")
            current_is_han = (
                current_start is not None
                and all(
                    is_han_component(character)
                    for character in source[
                        current_start:current_end]))
            piece_is_han = all(
                is_han_component(character)
                for character in source[start:end])
            continues = (
                label == "I"
                and current_start is not None
                and current_end == start
                and current_is_han == piece_is_han)
            if not continues:
                finish_current()
                current_start = start
            current_end = end
            current_scores.append(float(score))
        finish_current()
        return spans

    def _infer_units(self, units):
        """Run one inference pass for offset/source pairs."""
        decoded_units = [None] * len(units)
        scheduled = tuple(sorted(
            enumerate(units),
            key=lambda item: (len(item[1][1]), item[0]),
        ))
        for batch_start in range(0, len(scheduled), self.batch_size):
            indexed_batch = scheduled[
                batch_start:batch_start + self.batch_size]
            batch = tuple(
                unit
                for _original_index, unit in indexed_batch)
            sources = [source for _, source in batch]
            encoded = self._tokenizer(
                sources,
                add_special_tokens=True,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=False)
            offsets = encoded.pop("offset_mapping")
            model_inputs = {
                name: value.to(self._resolved_device)
                for name, value in encoded.items()
            }
            with self._torch.inference_mode():
                logits = self._model(**model_inputs).logits
                probabilities = self._torch.softmax(logits, dim=-1)
                scores, label_ids = probabilities.max(dim=-1)

            for row, (
                    original_index,
                    (source_offset, source),
            ) in enumerate(indexed_batch):
                decoded_units[original_index] = tuple(
                    self._decode_chunk(
                    source,
                    source_offset,
                    offsets[row].tolist(),
                    label_ids[row].tolist(),
                    scores[row].tolist(),
                    encoded["attention_mask"][row].tolist()))

        return tuple(
            unit if unit is not None else ()
            for unit in decoded_units)

    def _refine_long_han_spans(self, spans):
        """Reclassify suspicious long Han spans alone until stable.

        A long span is never split by a hard-coded boundary. It is replaced
        only when the same pinned model predicts more than one word with the
        unrelated surrounding sentence removed.
        """
        refined = list(spans)
        stable = set()

        while True:
            candidates = []
            for span in refined:
                key = (
                    span.start_offset,
                    span.end_offset,
                    span.surface,
                )
                if (
                        key not in stable
                        and len(span.surface) > CKIP_REFINEMENT_THRESHOLD
                        and all(
                            is_han_component(character)
                            for character in span.surface)):
                    candidates.append(span)
            if not candidates:
                break

            predictions = self._infer_units(tuple(
                (span.start_offset, span.surface)
                for span in candidates))
            replacements = {}
            for span, prediction in zip(
                    candidates,
                    predictions,
                    strict=True):
                key = (
                    span.start_offset,
                    span.end_offset,
                    span.surface,
                )
                if (
                        len(prediction) == 1
                        and prediction[0].start_offset == span.start_offset
                        and prediction[0].end_offset == span.end_offset
                        and prediction[0].surface == span.surface):
                    stable.add(key)
                    continue
                if (
                        len(prediction) < 2
                        or prediction[0].start_offset != span.start_offset
                        or prediction[-1].end_offset != span.end_offset
                        or "".join(
                            child.surface
                            for child in prediction) != span.surface):
                    raise ValueError(
                        "Isolated CKIP refinement did not preserve the "
                        "source span.")
                replacements[key] = prediction

            if not replacements:
                break
            next_refined = []
            for span in refined:
                key = (
                    span.start_offset,
                    span.end_offset,
                    span.surface,
                )
                next_refined.extend(replacements.get(key, (span,)))
            refined = next_refined

        return tuple(refined)

    def _tokenize_in_session(self, text):
        self._load()
        normalized = unicodedata.normalize("NFC", text)
        if normalized != text:
            raise ValueError(
                "Tokenizer input must already use NFC normalization.")
        units = split_model_units(text, self.max_characters)
        spans = self._refine_long_han_spans(
            tuple(
                span
                for unit in self._infer_units(units)
                for span in unit))

        previous_end = 0
        for span in spans:
            if (
                    span.start_offset < previous_end
                    or span.end_offset <= span.start_offset
                    or text[
                        span.start_offset:span.end_offset] != span.surface):
                raise ValueError(
                    "Tokenizer returned inconsistent source offsets.")
            previous_end = span.end_offset
        return tuple(spans)

    def tokenize(self, text):
        if self._execution_session_depth:
            return self._tokenize_in_session(text)
        with self.execution_session():
            return self._tokenize_in_session(text)
