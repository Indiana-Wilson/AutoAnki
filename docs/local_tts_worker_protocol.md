# AutoAnki local TTS worker protocol

Enhanced audio keeps each speech runtime in an isolated environment. AutoAnki
starts the configured worker for one operation, writes one UTF-8 JSON object
plus a newline to standard input, and expects exactly one JSON object on
standard output. Runtime logs belong on standard error. Production synthesis
uses bounded batches so a model is loaded once for several cache misses.

The current protocol identity is:

```json
{"protocol":"autoanki-local-tts","version":1}
```

Production route identities are:

| Requested language | Backend | Model | Voice |
| --- | --- | --- | --- |
| Japanese | `melotts_jp` | `myshell-ai/MeloTTS-Japanese` | `JP` |
| Chinese and Classical Chinese | `kokoro_82m_zh` | `hexgrad/Kokoro-82M-v1.1-zh` | `zm_010` |
| English | `fun_cosyvoice3_0_5b` | `Fun-CosyVoice3-0.5B` | `neutral-english` |
| French | `fun_cosyvoice3_0_5b` | `Fun-CosyVoice3-0.5B` | `neutral-french` |

Every request and response also carries the same `backend` and `operation`.
Workers must exit nonzero on a process-level failure. A handled error exits
zero and returns:

```json
{
  "protocol": "autoanki-local-tts",
  "version": 1,
  "backend": "fun_cosyvoice3_0_5b",
  "operation": "synthesize",
  "ok": false,
  "error": {"message": "Concise actionable explanation"}
}
```

## Status

The request identifies the expected model and makes the GPU requirement
explicit:

```json
{
  "protocol": "autoanki-local-tts",
  "version": 1,
  "backend": "fun_cosyvoice3_0_5b",
  "operation": "status",
  "model": {"id": "Fun-CosyVoice3-0.5B"},
  "execution": {
    "required_device": "cuda",
    "allow_cpu_fallback": false
  }
}
```

A ready response has this shape:

```json
{
  "protocol": "autoanki-local-tts",
  "version": 1,
  "backend": "fun_cosyvoice3_0_5b",
  "operation": "status",
  "ok": true,
  "runtime": {"available": true},
  "gpu": {"available": true, "device": "cuda", "name": "GPU name"},
  "model": {
    "id": "Fun-CosyVoice3-0.5B",
    "revision": "immutable model commit or local content digest"
  },
  "voices": {
    "neutral-english": {
      "revision_sha256": "64 lowercase hexadecimal characters"
    }
  }
}
```

The voice revision is the SHA-256 of every installed voice/reference asset
which can change synthesis. A worker should combine and hash multiple assets
deterministically. It must not return `available: true` until the model,
requested CUDA runtime, and revisioned voices are usable.

## Synthesis

AutoAnki sends normalized plain text, an immutable model and voice identity,
the requested settings and output format, and a host-controlled staging path.
The worker must use CUDA, must write only the requested file, and must not
substitute a CPU implementation. The single-item `synthesize` operation below
is retained for protocol compatibility:

```json
{
  "protocol": "autoanki-local-tts",
  "version": 1,
  "backend": "fun_cosyvoice3_0_5b",
  "operation": "synthesize",
  "model": {
    "id": "Fun-CosyVoice3-0.5B",
    "revision": "immutable revision"
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
    "path": "/shared/tts/staging/host-selected.wav"
  },
  "execution": {
    "required_device": "cuda",
    "allow_cpu_fallback": false
  }
}
```

A successful worker confirms the exact path, codec, encoder, and device:

```json
{
  "protocol": "autoanki-local-tts",
  "version": 1,
  "backend": "fun_cosyvoice3_0_5b",
  "operation": "synthesize",
  "ok": true,
  "artifact": {
    "path": "/shared/tts/staging/host-selected.wav",
    "device": "cuda",
    "codec": "wav",
    "encoder": "pcm_s16le"
  }
}
```

AutoAnki independently verifies that the file exists and is nonempty, hashes
it, and atomically publishes it to the shared content-addressed cache.

## Batched synthesis

Normal generation uses `synthesize_many`. Its `items` array contains the same
properties as a single synthesis request plus a unique content-addressed
`cache_key` on each item:

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
        "revision": "immutable revision"
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
        "path": "/shared/tts/staging/host-selected.wav"
      },
      "execution": {
        "required_device": "cuda",
        "allow_cpu_fallback": false
      }
    }
  ]
}
```

The response contains an `artifacts` array. Each artifact echoes its
`cache_key`, so response order is irrelevant. The worker validates the whole
batch before loading a model and removes every staged output if any item
fails. AutoAnki sends no more than 50 items to one subprocess even though the
protocol permits larger batches.

The parent holds `locks/orchestration.lock` from its final cache check through
atomic publication. A worker independently holds `locks/worker-gpu.lock`
while loading and using its model. These distinct shared locks serialize GPU
use across AutoAnki repositories without causing a parent/child self-deadlock.

## Runtime locations

The default shared root is `~/.local/share/autoanki/tts` (or the applicable
`XDG_DATA_HOME`). Model downloads use the normal Hugging Face cache
(`HF_HOME`, `XDG_CACHE_HOME/huggingface`, or `~/.cache/huggingface`).

The default isolated worker locations are:

- `runtimes/fun_cosyvoice3_0_5b/bin/python` with
  `runtimes/fun_cosyvoice3_0_5b/autoanki_worker.py`
- `runtimes/melotts_jp/bin/python` with
  `runtimes/melotts_jp/autoanki_worker.py`
- `runtimes/kokoro_82m_zh/bin/python` with
  `runtimes/kokoro_82m_zh/autoanki_worker.py`
- optional comparison runtime:
  `runtimes/style_bert_vits2_jp_extra/bin/python` with
  `runtimes/style_bert_vits2_jp_extra/autoanki_worker.py`

On Windows, `Scripts/python.exe` is used. The commands can be overridden with
`AUTOANKI_COSYVOICE3_WORKER`, `AUTOANKI_MELOTTS_JP_WORKER`,
`AUTOANKI_KOKORO_82M_ZH_WORKER`, and the optional
`AUTOANKI_STYLE_BERT_VITS2_WORKER`. An override is parsed as an argument
vector and is never executed through a shell.
