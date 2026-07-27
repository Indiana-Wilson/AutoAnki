# Installing AutoAnki's local TTS runtimes

AutoAnki keeps speech dependencies outside its own virtual environment. The
default shared directory is:

```text
~/.local/share/autoanki/tts
├── artifacts/                  # content-addressed generated audio
├── models/                     # immutable Hugging Face snapshots
├── runtimes/
│   ├── style_bert_vits2_jp_extra/
│   └── fun_cosyvoice3_0_5b/
├── sources/                    # pinned CosyVoice checkout and submodules
└── staging/                    # host-controlled temporary output
```

This location can be changed with `AUTOANKI_TTS_HOME` or the installer's
`--root` option. Hugging Face downloads continue to use the shared `HF_HOME`
cache.

## Inspect before installing

The plan command performs no installation or model download:

```bash
python3 scripts/install_local_tts.py --backend all --dry-run
```

Style-Bert-VITS2 supports the installer's Python 3.10–3.12 path. The upstream
CosyVoice instructions specify Python 3.10, so that backend deliberately
requires a Python 3.10 interpreter:

```bash
python3 scripts/install_local_tts.py \
  --backend style \
  --style-python /path/to/python3.11

python3 scripts/install_local_tts.py \
  --backend cosy \
  --cosy-python /path/to/python3.10
```

On a machine without system Python 3.10, a uv-managed interpreter can remain
inside the same shared root. For example, this installation uses the stable
path:

```text
~/.local/share/autoanki/tts/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10
```

Pass that path with both `--style-python` and `--cosy-python`. Keeping uv's
binary, Python installation, and cache below the shared TTS root makes the
runtime discoverable to other repositories without changing the system Python
or AutoAnki's virtual environment.

The installer:

1. creates a venv directly under the shared runtime directory;
2. installs a CUDA PyTorch wheel there, without touching AutoAnki's `.venv`;
3. verifies that PyTorch can allocate a tensor on the NVIDIA GPU;
4. installs a pinned backend revision;
5. downloads an immutable model snapshot and installed voice/reference asset;
6. writes an installation manifest in the `testing` state;
7. loads the model once and synthesizes a small GPU-only batch; and
8. changes the manifest to `ready` only after that audio smoke test succeeds.

There is no CPU fallback. A missing CUDA runtime, driver mismatch, model
load problem, or empty output fails installation.
The short WAV files are retained under
`diagnostics/installation-smoke/<backend>-<timestamp>` for a human
intelligibility and neutrality check; merely detecting a non-empty waveform
cannot prove speech quality.

If another trusted process temporarily occupies most GPU memory, prepare the
isolated environments and immutable snapshots without attempting to load a TTS
model:

```bash
python3 scripts/install_local_tts.py \
  --backend all \
  --defer-smoke \
  --style-python /path/to/python3.10 \
  --cosy-python /path/to/python3.10
```

Prepared manifests remain in the non-ready `testing` state. Once VRAM is
available, promote them only by passing real CUDA audio smoke tests:

```bash
python3 scripts/install_local_tts.py \
  --backend all \
  --verify-existing
```

If only repository worker code changes, refresh the two small worker files and
their composite cache identities without reinstalling models or environments:

```bash
python3 scripts/install_local_tts.py \
  --backend all \
  --refresh-workers
```

This returns each selected runtime to `testing`, revalidates CUDA
torch/torchaudio, reconciles both environment and worker fingerprints, and
therefore requires another `--verify-existing` smoke before use.

## Replacing or rolling back a runtime

The installer refuses to overwrite an existing runtime. `--force` moves the
old runtime to a timestamped sibling before creating its replacement:

```bash
python3 scripts/install_local_tts.py \
  --backend style \
  --force
```

If replacement fails, the incomplete runtime is retained with a `.failed-*`
name and the former runtime is restored. Old model snapshots are immutable and
are not deleted.

## Worker protocol and batching

The worker is a one-process JSON-lines program. A status process does not load
the model. Audio generation uses `synthesize_many`, allowing one process to
load a backend once and synthesize all items sequentially:

```json
{
  "protocol": "autoanki-local-tts",
  "version": 1,
  "backend": "fun_cosyvoice3_0_5b",
  "operation": "synthesize_many",
  "items": [
    {
      "cache_key": "64 lowercase hexadecimal characters",
      "model": {
        "id": "Fun-CosyVoice3-0.5B",
        "revision": "installed immutable identity"
      },
      "input": {
        "text": "Bonjour",
        "language": "french",
        "role": "word"
      },
      "voice": {
        "id": "neutral-french",
        "revision_sha256": "64 lowercase hexadecimal characters",
        "reference_path": null,
        "reference_sha256": null
      },
      "settings": {
        "emotion": "neutral",
        "speed": 1.0,
        "pitch": 0.0,
        "energy": 1.0,
        "seed": 0
      },
      "output": {
        "codec": "wav",
        "encoder": "pcm_s16le",
        "sample_rate_hz": 24000,
        "channels": 1,
        "path": "/shared/root/staging/example.wav"
      },
      "execution": {
        "required_device": "cuda",
        "allow_cpu_fallback": false
      }
    }
  ]
}
```

Each returned artifact echoes its `cache_key`, so callers need not depend on
response ordering. Workers validate the entire batch before loading a model.
If an item later fails, all staged outputs from that batch are removed.
Every worker also holds `locks/worker-gpu.lock` across model loading and the
entire batch, preventing unrelated repositories from loading two large models
onto the same GPU at once. AutoAnki separately holds
`locks/orchestration.lock` through shared-cache publication; the distinct lock
names avoid a parent/child self-deadlock.

## Backends and installed voices

- Japanese uses Style-Bert-VITS2 JP-Extra with the official
  `jvnv-F1-jp` model and its `Neutral` style.
- English, French, and Chinese use
  `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` with the official CosyVoice
  zero-shot reference asset. The three stable AutoAnki voice identities share
  that revisioned reference.

The CosyVoice environment records two deliberate compatibility overrides.
`openai-whisper==20250625` preserves the pinned release's tokenizer/log-Mel
behavior while supporting current Triton, and
`onnxruntime-gpu==1.23.2` supplies the CUDA 12/cuDNN 9 runtime needed by the
speech-tokenizer graph. Its packaged CUDA libraries are explicitly preloaded
before PyTorch's CUDA 13 context, and the worker rejects a session that falls
back from `CUDAExecutionProvider`. Reference WAV decoding uses SoundFile
instead of TorchCodec, so no optional FFmpeg/NPP installation or CPU model
fallback is hidden in synthesis.

Classical Chinese is routed to the Chinese voice and is therefore pronounced
as modern Mandarin; the worker does not claim a reconstructed historical
pronunciation.

The Style-Bert-VITS2 source is AGPL-3.0 and its installed `jvnv` voice assets
are CC-BY-SA-4.0. CosyVoice and its model are Apache-2.0. The installer writes
the source URLs and license identifiers to `THIRD_PARTY_NOTICES.md` in the
shared TTS root.
