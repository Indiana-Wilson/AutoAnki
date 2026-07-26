# Archived shared local Ollama model notes

AutoAnki's From Source generator can use two local, schema-constrained models
without sending card content to OpenAI:

| AutoAnki model ID | Shared Ollama tag | Quantization | Approximate download |
| --- | --- | --- | ---: |
| `ollama/gemma4:12b` | `gemma4:12b` | Q4_K_M | 7.6 GB |
| `ollama/qwen3:14b` | `qwen3:14b` | Q4_K_M | 9.3 GB |

The tags are the official Ollama packages for
[Gemma 4 12B](https://ollama.com/library/gemma4:12b) and
[Qwen3 14B](https://ollama.com/library/qwen3:14b). Qwen3's post-trained
14B model supports both thinking and non-thinking operation; there is no
separate Ollama `qwen3:14b-instruct` tag. Do not substitute an uncensored,
abliterated, community-converted, or lower-precision model under either
supported name.

## One service and model store per user

The configured Linux workstation runs Ollama as a user service:

```text
executable   ~/.local/bin/ollama
service      ~/.config/systemd/user/ollama.service
model store  ~/.ollama/models
API          http://127.0.0.1:11434
```

This is deliberately outside the repository and its Python virtual
environment. Every AutoAnki checkout, other repository, terminal, and Codex
agent running as the same operating-system user sees the same model store and
localhost API. A different operating-system user, container, or remote machine
does not automatically share it.

Do not put GGUF files in a repository, set a per-project `OLLAMA_MODELS`, or
run a second Ollama server on another port. Those approaches waste disk and
make model identity difficult to audit.

## Service configuration

Install Ollama once using its
[official Linux instructions](https://docs.ollama.com/linux). The current
user-service configuration should retain loopback-only access and local-only
inference, and should constrain memory use:

```ini
[Unit]
Description=Ollama local model server
After=network-online.target

[Service]
ExecStart=%h/.local/bin/ollama serve
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_NO_CLOUD=1"
Environment="OLLAMA_CONTEXT_LENGTH=8192"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

After installing or changing the unit:

```bash
systemctl --user daemon-reload
systemctl --user enable --now ollama
systemctl --user status ollama --no-pager
curl --fail http://127.0.0.1:11434/api/version
```

Keep `OLLAMA_HOST` on `127.0.0.1`. Ollama has no application-level
authentication in this setup and must not be exposed directly to a LAN or the
public internet. `OLLAMA_NO_CLOUD=1` prevents accidental cloud inference; it
does not prevent explicit model downloads.

Flash Attention is required for quantized KV cache support. Ollama documents
`q8_0` KV cache as using about half the memory of `f16`, normally with
negligible quality loss. This changes the transient context cache, not the
Q4_K_M model weights.

## Download and verify the exact models

Pull each tag once from any directory:

```bash
ollama pull gemma4:12b
ollama pull qwen3:14b
ollama list
ollama show gemma4:12b
ollama show qwen3:14b
```

The pull is shared by all repositories for the user. `ollama show` should
report `quantization Q4_K_M` for both tags; AutoAnki refuses to authorize a
different quantization under either supported name. The two weights require
roughly 17 GB combined, in addition to temporary download headroom and
existing models.

AutoAnki records the exact Ollama tag digest, runtime version, inference
policy, and base URL when a job is authorized. It checks that snapshot again
before generation and retry. If `ollama pull` changes a tag after
authorization, AutoAnki refuses to silently resume with different weights;
recalculate the estimate and create a new job.

## Workstation limits

The checked-in limits target the current workstation:

- NVIDIA RTX 5050 Laptop GPU with approximately 8 GB VRAM;
- approximately 22 GiB system RAM;
- no swap;
- one model and one inference request at a time;
- an 8,192-token context window;
- five source terms per request by default.

Gemma 4 12B is close to the available VRAM limit and may offload a small
portion to the CPU. Qwen3 14B is larger than VRAM and necessarily uses hybrid
CPU/GPU inference. It will be slower, but Q4_K_M is retained to protect
multilingual and semantic quality. Do not silently replace it with Q3_K_M to
make it fit.

AutoAnki deliberately leaves Ollama's `num_gpu` option unset. Ollama therefore
places as many layers on the GPU as fit and keeps the remainder in system RAM.
Forcing full GPU placement would exceed the 8 GB GPU on Qwen3 and can fail with
an out-of-memory error; hybrid CPU/GPU placement is still GPU acceleration.
After every completed local response, AutoAnki queries Ollama's official
`/api/ps` endpoint for the exact selected tag. It also checks that the chat
response names that model, the running-model digest is well formed, and the
allocated context matches the request. It rejects the response if any of
those checks fail, if the model is absent from the running-model list, or if
`size_vram` is not a positive integer no larger than the total allocation.
The local response metadata includes the loaded digest, total allocation
size, VRAM allocation, context length, and the computed VRAM fraction.

Large evaluations should increase the number of sequential chunks, not the
number of words in a chunk. For example, test the first 100 source terms as
about twenty five-word requests. A single 100-word response is likely to
exhaust the context window or system memory even though the model advertises a
much larger theoretical context.

No swap makes an out-of-memory event capable of terminating AutoAnki or other
desktop processes. Before a large run, close unnecessary GPU-heavy
applications and confirm available memory. Do not load both supported models
at once.

## Running from AutoAnki

1. Start Ollama and confirm that the selected tag appears in `ollama list`.
2. Open **Generate → From Source → Generate Deck**.
3. Select **Gemma 4 12B Instruct · local Q4_K_M** or
   **Qwen3 14B Instruct · local Q4_K_M**.
4. Retain the suggested five words per request and concurrency of one for the
   first run.
5. Local generation uses the compact-v10 no-reasoning path. The control is
   locked so thinking tokens cannot consume the small context window or
   interfere with schema-only output.
6. Optionally select **Automatic repair**. Local models may make at most five
   automatic validation follow-ups per failed base request, starting with the
   smallest safe scope and escalating to a complete failed-chunk retry only
   when narrower repair is impossible.
7. Calculate the estimate, authorize local generation, and inspect validation
   failures before importing.

Local models support only direct Standard processing. Economy Batch and
OpenAI web search are disabled. The displayed provider price is zero; it does
not estimate electricity or hardware wear. An OpenAI API key is not required
for local source generation.

Both models receive the same compact prompt, strict JSON Schema, local
validation, translation memory, conservative deterministic repairs, job
persistence, and Anki packaging as the paid model. The local adapter sends the
schema through Ollama's structured-output `format` field with `think: false`.

## Using the shared service from another repository

The health and inventory endpoints require no AutoAnki imports:

```bash
curl --fail http://127.0.0.1:11434/api/version
curl --fail http://127.0.0.1:11434/api/tags
```

A minimal structured-output smoke test is:

```bash
curl --fail http://127.0.0.1:11434/api/chat \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "qwen3:14b",
    "messages": [{"role": "user", "content": "Return the number one."}],
    "stream": false,
    "think": false,
    "format": {
      "type": "object",
      "properties": {"value": {"type": "integer"}},
      "required": ["value"],
      "additionalProperties": false
    },
    "options": {"num_ctx": 8192, "temperature": 0}
  }'
```

Other applications should use the same base URL and exact tags. AutoAnki
accepts `AUTOANKI_OLLAMA_BASE_URL` for a different HTTP(S) server URL, but the
supported and safest configuration is the local loopback service.

The server is shared execution infrastructure as well as a shared model
store. Run only one local generation or audit job at a time across AutoAnki,
other repositories, and Codex agents. A concurrent request for another model
can evict the just-used runner before AutoAnki reads `/api/ps`; in that race,
the GPU guard deliberately fails closed and the response is not accepted.

## Monitoring and recovery

Use these read-only checks while a model is loaded:

```bash
ollama ps
nvidia-smi
free -h
journalctl --user -u ollama.service --no-pager -n 100
```

`ollama ps` reports whether layers are on the GPU or CPU and the allocated
context. Hybrid placement is expected for Qwen3 14B. AutoAnki fails closed
instead of accepting a CPU-only response, but it does not require 100% GPU
placement.

If AutoAnki reports that Ollama is unavailable:

```bash
systemctl --user restart ollama
curl --fail http://127.0.0.1:11434/api/version
```

If a model is missing, pull the exact supported tag. If generation is killed
or reports insufficient memory, verify that only one model/request is active,
close competing workloads, retain the 8,192-token context, and reduce the
request chunk size. Do not repair an out-of-memory problem by raising the
context window or parallelism.
