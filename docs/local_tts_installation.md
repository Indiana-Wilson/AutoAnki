# Installing AutoAnki's local TTS runtimes

AutoAnki keeps speech dependencies outside its own virtual environment. The
default shared directory is:

```text
~/.local/share/autoanki/tts
├── artifacts/                  # content-addressed generated audio
├── models/                     # immutable Hugging Face snapshots
├── runtimes/
│   ├── fun_cosyvoice3_0_5b/    # production English and French
│   ├── melotts_jp/             # production Japanese
│   ├── kokoro_82m_zh/          # production Chinese
│   └── style_bert_vits2_jp_extra/ # optional legacy comparison
├── sources/                    # pinned CosyVoice and MeloTTS checkouts
└── staging/                    # host-controlled temporary output
```

This location can be changed with `AUTOANKI_TTS_HOME` or the installer's
`--root` option. Hugging Face downloads continue to use the shared `HF_HOME`
cache.

## Inspect before installing

The plan command performs no installation or model download:

```bash
python3 scripts/install_local_tts.py --backend production --dry-run
```

`production` selects CosyVoice, MeloTTS, and Kokoro. `all` additionally
installs the optional Style-Bert comparison runtime, which has a substantial
separate disk cost.

The active production routes use CosyVoice on Python 3.10, MeloTTS on Python
3.10, and Kokoro on Python 3.10–3.12. Style-Bert-VITS2 remains available as
an optional comparison backend on Python 3.10–3.12:

```bash
python3 scripts/install_local_tts.py \
  --backend cosy \
  --cosy-python /path/to/python3.10

python3 scripts/install_local_tts.py \
  --backend melo \
  --melo-python /path/to/python3.10

python3 scripts/install_local_tts.py \
  --backend kokoro \
  --kokoro-python /path/to/python3.11

python3 scripts/install_local_tts.py \
  --backend style \
  --style-python /path/to/python3.11
```

On a machine without system Python 3.10, a uv-managed interpreter can remain
inside the same shared root. For example, this installation uses the stable
path:

```text
~/.local/share/autoanki/tts/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10
```

Pass that path with `--cosy-python`, `--melo-python`, and `--kokoro-python`
when installing the production set. Keeping uv's binary, Python installation,
and cache below the shared TTS root makes the runtime discoverable to other
repositories without changing the system Python or AutoAnki's virtual
environment.

The installer:

1. creates a venv directly under the shared runtime directory;
2. installs a CUDA PyTorch wheel there, without touching AutoAnki's `.venv`;
3. verifies that PyTorch can allocate a tensor on the NVIDIA GPU;
4. installs exact synthesis dependency versions and a pinned backend revision;
5. downloads an immutable model snapshot and installed voice/reference asset;
6. writes an installation manifest in the `testing` state;
7. loads the model once and synthesizes a small GPU-only batch; and
8. changes the manifest to `ready` only after that audio smoke test succeeds.

There is no CPU fallback. A missing CUDA runtime, driver mismatch, model
load problem, or empty output fails installation.
The manifest records the complete installed-environment fingerprint. MeloTTS
also records and rechecks a content hash of its imported source package, so an
edited checkout is refused instead of generating different audio under an old
cache identity.
The short WAV files are retained under
`diagnostics/installation-smoke/<backend>-<timestamp>` for a human
intelligibility and neutrality check; merely detecting a non-empty waveform
cannot prove speech quality.

If another trusted process temporarily occupies most GPU memory, prepare the
isolated environments and immutable snapshots without attempting to load a TTS
model:

```bash
python3 scripts/install_local_tts.py \
  --backend production \
  --defer-smoke \
  --cosy-python /path/to/python3.10 \
  --melo-python /path/to/python3.10 \
  --kokoro-python /path/to/python3.10
```

Prepared manifests remain in the non-ready `testing` state. Once VRAM is
available, promote them only by passing real CUDA audio smoke tests:

```bash
python3 scripts/install_local_tts.py \
  --backend production \
  --verify-existing
```

If only repository worker code changes, refresh the selected small worker
files and their composite cache identities without reinstalling models or
environments:

```bash
python3 scripts/install_local_tts.py \
  --backend production \
  --refresh-workers
```

This static maintenance operation returns each selected runtime to `testing`,
preserves the already-recorded environment fingerprint, and reconciles the
worker fingerprint without importing Torch or touching CUDA. It therefore
still requires another `--verify-existing` smoke before use.

## Replacing or rolling back a runtime

The installer refuses to overwrite an existing runtime. `--force` moves the
old runtime to a timestamped sibling before creating its replacement:

```bash
python3 scripts/install_local_tts.py \
  --backend melo \
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
AutoAnki's subprocess launcher holds the host-wide cooperative lease at
`~/.local/share/autoanki/tts/locks/worker-gpu.lock` from before starting a
worker until after that one-shot process exits. This covers model loading, the
complete batch, interpreter/model teardown, and status probes. An absolute
`LOCAL_LLM_GPU_LOCK_PATH` moves the parent-held lease when every participating
application receives the same value. The smoke-qualified worker still takes
its historical child-side lock, so the launcher gives it a distinct delegated
lock path to avoid recursively acquiring the parent-held lease. Direct worker
and installer invocations continue to take the host-wide default themselves.
On POSIX the parent also passes the already-locked descriptor into the worker,
so an abruptly terminated parent cannot release the host lease while its
orphaned worker is still alive.
AutoAnki separately holds `locks/orchestration.lock` through shared-cache
publication. The GPU lease only serializes cooperating workloads; it does not
require swap or impose a fixed host-RAM threshold.

## Voice inventory and listening samples

The Japanese MeloTTS model has one installed speaker: `JP`. The Chinese
Kokoro model contains the explicitly selected `zm_010` voice file. Production
routing does not silently substitute another speaker.

CosyVoice is a zero-shot voice-cloning model rather than a finite catalogue
of built-in speakers. AutoAnki's `neutral-english` and `neutral-french`
identities use the same revision-pinned official reference voice.

The selected voices and candidate demos are retained in
`output/evaluations/tts_voice_evaluation_2026-07-27/`. The older helper below
recreates the pre-selection Style-Bert/CosyVoice comparison only:

```bash
python3 scripts/generate_tts_voice_samples.py
```

That script does not represent the current Japanese or Chinese routes and
does not change production voice configuration.

## Backends and installed voices

- Japanese uses `myshell-ai/MeloTTS-Japanese` with speaker `JP`.
- Chinese, including Classical Chinese variants, uses
  `hexgrad/Kokoro-82M-v1.1-zh` with voice `zm_010`.
- English and French use
  `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` with the official CosyVoice
  zero-shot reference asset. The two stable AutoAnki voice identities share
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

Classical Chinese is routed to Kokoro's Chinese voice and is therefore
pronounced as modern Mandarin; the worker does not claim a reconstructed
historical pronunciation.

MeloTTS and its Japanese model are MIT licensed. Kokoro and its selected
Chinese model are Apache-2.0, as are CosyVoice and its model. The optional
Style-Bert source is AGPL-3.0 and its `jvnv` assets are CC-BY-SA-4.0. The
installer writes source URLs and license identifiers to
`THIRD_PARTY_NOTICES.md` in the shared TTS root.
