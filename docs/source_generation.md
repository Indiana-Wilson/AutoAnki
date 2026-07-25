# From Source generation

The Generate tab has two modes:

- **Manual Input** keeps the ordinary text-box workflow.
- **From Source** creates vocabulary cards from a processed, inspectable
  source while retaining progress per OpenAI request.

From Source contains five pages:

1. **Generate Deck** selects the source, its language/register, context,
   request size, concurrency, optional web search, cost estimate, and optional
   Anki vocabulary exclusion.
2. **Inspect Source** pages through the first-occurrence vocabulary and shows
   each candidate's section and previous/current/next sentence metadata.
3. **Prepare File** extracts and tokenizes a local PDF, UTF-8 TXT, or Markdown
   file without generating cards.
4. **Codex Retrieve** gives an explicitly authorized Codex CLI process a
   source-retrieval description, then prepares the retrieved files.
5. **Jobs & Failures** shows each saved request, its attempts, and the final
   package/import step. Each deck-generation job is an expandable group, with
   its request chunks in order and its Deck packaging/import row last.
   **Inspect selected stage** separates the exact request from the retained
   response in two tabs; it inspects one chunk, not the whole job.

Daodejing and Journey to the West are the built-in presets. Locally prepared
files appear in the same source selector after preparation succeeds. A preset
name in the selector does not replace its processed corpus artifacts: the
corresponding audited build must be present under `output/corpora/`.
If two prepared sources have the same title, the selector appends each
source's stable key to make the rows unambiguous. That suffix is a UI label
only: the retained source title, prompt metadata, and
`Vocabulary from <source title>` deck name are not changed.

## Before generating

Card directions, response fields, and response languages come from the
selected source language's settings in **Card setup**. Changing any of those
choices changes the source cost estimate. The source workflow then adds these
controls:

- **Source language / register**, prefilled from prepared-source metadata
- **Context per word**
  - No source context
  - Current sentence
  - Current, previous, and next sentence
  - Complete source span covered by the chunk
- **Use source for example sentences**, available when source context and the
  Context card direction are enabled
- optional **Generate only from the first N token occurrences**, measured in
  running source words before vocabulary deduplication
- **Words per OpenAI request**, which is the generation chunk size
- **Request protocol**
  - Compact v9, the default fixed-schema protocol
  - Legacy v8, a one-click rollback that preserves its grouped behavior
- **Reasoning**
  - None, the default economy-oriented setting
  - Low, an optional quality/cost trade-off
- **Processing**
  - Standard synchronous requests
  - Economy, an asynchronous OpenAI Batch at half token pricing
- **Max parallel requests**
- **Minimum start stagger**
- optional web search when a meaning remains unclear
- an optional exact Anki vocabulary-exclusion selector

Changing the generation chunk size does not scrape or tokenize the source
again. It repartitions the saved first-occurrence vocabulary. Smaller chunks
repeat the prompt and JSON schema more often; larger chunks reduce that
overhead but create larger inputs and outputs. AutoAnki blocks generation when
the estimate indicates that one request may exceed the configured model
limits.

The source-prefix option is a true text cutoff, not a limit on unique cards.
For example, if the first 100 running words contain 63 distinct normalized
forms, at most those 63 forms are candidates. Repeated words still consume
positions in the 100-word prefix. AutoAnki first selects the vocabulary whose
first occurrence falls inside that prefix, then applies learned-word
exclusions, and only then partitions the remainder into OpenAI requests. The
estimate distinguishes the requested/actual prefix length, unique forms in
the prefix, forms omitted because they first occur later, learned-word
exclusions, and final candidate count. The cutoff and all of those counts are
retained in the saved plan.

The defaults are 30 words per request, compact v9, no reasoning, Standard
processing, eight concurrent workers, and a 100 ms minimum interval between
request starts. OpenAI limits vary by account,
project, model, and usage tier, so there is no universal safe maximum worker
count. Eight is the higher conservative startup value; the shared coordinator
honours server `Retry-After` instructions and automatically backs off all
not-yet-started workers after a rate-limit response.

The displayed cost is an estimate in Australian dollars, not a quote. OpenAI
prices are calculated in USD, then converted using the frozen RBA AUD/USD
rate of 0.6975 published at 4:00 pm on 2026-07-24. The configured pricing
profile is `gpt-5.4-mini` at US$0.75 per million uncached input tokens,
US$0.075 per million cached input tokens, and US$4.50 per million output
tokens, labelled **standard pricing published 2026-07-24**. Economy uses the
documented Batch rates of US$0.375, US$0.0375, and US$2.25 respectively.
Freezing the exchange rate makes the estimate reproducible between
authorization and job creation. It does not include card-issuer
currency-conversion fees. The configured request guards are a 400,000-token
context window and 128,000 maximum output tokens. These are application
constants, not a live price or model-metadata lookup; check the
[current official model page](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
before relying on them at a later date.

The estimate includes selected Card setup fields and examples, the separate
contextual and additional-sense response shapes, one translation per shared
source context, exact serialized payload/schema sizes, prompt repetition, and
the number of words left after exclusion. Its headline is the modelled central
value; a labelled low/high range remains visible rather than being presented
as the likely bill. For cache-eligible requests, the central estimate treats
the first shared prefix as cold and later identical prefixes as cached; the
high estimate keeps every input cold. Token counts use a deterministic
character-class approximation. Estimation never calls OpenAI. If Anki
exclusion is enabled, calculating the estimate makes a read-only AnkiConnect
query or reuses the matching read-only result for up to 30 seconds.

Every received paid response separately records provider-reported input,
cached-input, output, reasoning, and total tokens before local validation.
Jobs & Failures prices that retained usage at the job's frozen Standard or
Economy profile and shows the dated USD/AUD basis. Reasoning is already a
subset of output and cached input is already a subset of input, so neither is
double-counted. Recorded web-search calls are added separately. Recovery may
retain the same provider response in more than one attempt directory; repeated
nonempty provider response IDs are counted only once in the displayed usage.

When web search is enabled, the model is told to use it only if the retained
context and its knowledge are insufficient. Each response is capped at one
built-in tool call. Its prompt directs it to identify all unclear terms first
and use that call as a consolidated search action covering the whole unclear
set, including multiple queries in the action when supported. The low end
assumes no search; the high end assumes one US$0.01 search call per request
plus 8,000 retrieved search-content tokens per call billed at the model's
input-token rate. The retrieved-token allowance is a conservative budgeting
assumption, not a promise about actual usage or search quality.

Changing an estimate-relevant option clears the paid-authorization checkbox.
No source card request is sent until the estimate is available and
**I authorise the paid OpenAI requests shown in this estimate** is ticked.

## Context sharing

Every saved candidate retains its first occurrence, section, paragraph, and
previous/current/next sentence metadata. A request contains one ordered word
record per candidate and a `context_id` that points to a separate context
record. Current jobs also persist a bounded `occurrence_locator` beside that
word. Its visibly marked target and adjacent excerpts make repeated spellings
unambiguous to the model without duplicating an entire long context.

Identical sentence contexts are serialized once per request. Overlapping
three-sentence windows are unioned into one interval, so words sharing any
part of that context point into the same combined unit. Complete-span mode
creates one continuous source interval from the earliest first-occurrence
sentence to the latest in that request, including intervening sentences that
contain no newly selected word. At the beginning or end of a source, a
missing previous or next sentence is simply omitted. Context cannot be
deduplicated across
different API requests because each request must be self-contained.

When **Use source for example sentences** is enabled, compact v9 uses one
fixed strict schema for every request. Its `term_results` array contains an
integer `rank`, one lexical-only `contextual_sense`, and an
`additional_senses` array. A second `source_context_translations` array carries
one `{context_id, translation}` item per shared passage. The immutable local
chunk, not a chunk-specific schema or model-echoed term, defines the exact
allowed rank and context-ID sets. AutoAnki rejects omissions, duplicates,
unknown identities, wrong ordering, or unsafe fields before packaging.

The v9 request payload omits redundant chunk/range/offset structures, keeps
only rank, term, context ID, a bounded marked occurrence when needed, and each
deduplicated context. Audited ambiguous Wang Bi occurrences carry short
term-local quality hints instead of repeating the old multi-page checks in
every request. Their unequivocal translation, reading, and grammatical role
are also checked locally, so a contradictory contextual card is retained for
review/repair rather than entering the deck. The fixed prompt/schema prefix is
eligible for automatic prompt caching. Legacy v8 remains selectable and
continues to build one exact, rank-keyed schema per chunk; existing v8 request
bytes and saved contracts are not migrated.

Each v8 or v9 rank
contains one lexical-only `contextual_sense` and an `additional_senses` array.
Additional senses contain four generated sentences and four aligned
translations as fixed-length JSON arrays. Sense objects omit the
source-language term, which AutoAnki restores from the immutable rank-to-word
mapping instead of trusting model-echoed spelling.

The root source-context collection translates each exact shared context once.
AutoAnki validates the exact root, rank, group, and context-ID keys, inserts
the exact context record as each contextual card's sole example, then merges
the contextual and additional senses in source-rank order. The inserted
context may contain one sentence, several sentences, or a larger text block.
A literal `|` already present in source text remains part of that one example
rather than becoming a delimiter. With the option off, every sense continues
to require four generated examples.

Every new Context card also stores an English translation positionally aligned
with each example. The back of the card shows the translation of the randomly
selected front example above the definition fields. Generated senses require
four translations in the same order as their four examples. A retained source
passage is one example and therefore requires one complete translation of that
whole passage. The exact request contract remains frozen with its job.
Obsolete v4-v7 source-context jobs remain inspectable, but cannot be retried or
finalized: their older shapes cannot structurally require every requested
rank, and earlier versions also had ambiguous sense grouping or occurrence
selection. Create a new v9 job instead. Older non-source-context contracts
retain their frozen behavior.

For model-generated examples, the model must wrap the usage that expresses the
card's particular sense in `<strong>...</strong>`. Grouped v8 and compact v9
source requests require the exact requested spelling once per example, inside
one exact strong span. Validation reports both repeated emphasized uses and
any remaining unmarked literal occurrence, including an occurrence embedded
in a compound. The v8 per-rank schema can require the exact marked surface for
one-character CJK terms. The fixed v9 schema requires a nonempty strong span
in every generated example; its immutable rank-to-term mapping and local
validation then enforce the exact requested spelling and reject any additional
unmarked occurrence.

Where an audited source-specific construction has one unequivocal analysis,
the v8 per-rank schema may freeze that occurrence's lexical constants. Compact
v9 instead carries a short term-local quality hint and repeats only concise
immutable facts in local validation. The Wang Bi opening's `天地之始`, for
example, constrains contextual `始` to `beginning; origin` as a noun; retained
raw output cannot bypass that local check.
Older contracts may mark an appropriate inflected form rather than merely
repeating the response term. Validation also rejects source-language text
copied into an English translation, including exact sentence copies and
Han-only values, and applies the same guard to the shared context translations.
Packaging preserves validated strong spans while escaping all other HTML.

Source passages are inserted only after the response, so the model cannot mark
them. AutoAnki uses the audited zero-based `[start, end)` occurrence span stored
with each requested word to highlight exactly the occurrence that selected the
contextual sense, even when the same surface form appears elsewhere in the
retained passage. A current source-context job with a missing, partial, or
inconsistent span fails non-overrideable validation and must be recreated from
the current audited corpus build. The locator is prompt guidance only; the
audited offsets remain authoritative for highlighting.

## Anki exclusion

In **Card setup**, enable the learned-word filter on the language page you are
configuring, then add one or more sources:

1. the exact Anki deck;
2. the exact note type represented by notes in that deck; and
3. the exact field containing the learned word.

AutoAnki uses AnkiConnect to discover the note types represented in the
selected deck and the fields belonging to the selected note type. The union
of every configured row is excluded. Adding a row copies the preceding deck,
note type, and field to make closely related selections quicker.

Card template is intentionally absent from the normal filter UI. A note type
defines stored fields; its card templates only decide how one note produces
one or more cards. Legacy retained jobs with an old card-template restriction
remain readable for recovery.

AutoAnki reads only matching notes. Field HTML is reduced to visible text, HTML
entities are decoded, whitespace is collapsed, and Unicode is normalized to
NFC. Comparison is otherwise exact: AutoAnki does not case-fold, stem,
lemmatize, or split a field containing several words.

If every source candidate is already present, AutoAnki displays a notice and
sends no OpenAI request, writes no `.apkg`, and imports no deck.

Anki must be reachable for the exclusion count. This lookup may start local
Anki and wait for its profile/startup sync, but it does not change the
collection. An estimate may use the short-lived read-only cache described
above. After paid generation is confirmed, AutoAnki waits for Anki again and
forces a fresh vocabulary read immediately before it freezes the paid source
plan. It therefore does not rely on the estimate's cached collection view for
the actual request set. The Manual Input exclusion path likewise forces a
fresh read before any card request.

## Requests, Economy Batch, concurrency, and rate limits

Standard generation creates one independent OpenAI request unit per source
chunk; each attempt for that unit calls the Responses API. Request workers use
a bounded thread pool; the default is eight workers with at least 100 ms between request
starts. This parallelism reduces time spent waiting on network responses. A
larger number is not automatically better: the account/model
request-per-minute and token-per-minute limits still apply.

The coordinator treats connection/time-out failures, HTTP 429, and HTTP 5xx
as transient. It uses bounded exponential backoff with positive jitter and
honors `Retry-After`, `retry-after-ms`, and OpenAI request/token reset headers.
A server cooldown is applied to the shared request-start gate, slowing queued
workers as well as the worker that received the limit. The OpenAI SDK's own
automatic retries are disabled so this persisted retry policy is the only
automatic layer.

Only those transient failures are retried automatically, up to the saved
limit. A refusal, incomplete output, malformed JSON, unexpected/missing word,
or other schema/semantic validation error is retained as
`invalid_response` and stops. **Retry selected** and **Retry all failed** both
show a fresh paid-action confirmation. Successful chunks are never included
in those retries. **Retry selected** resets and runs only the selected failed
request rows; its remaining-failure notice is scoped to the selected parent
jobs rather than unrelated old jobs.

Economy writes the same frozen `/v1/responses` request bodies to a JSONL file,
uploads it with `purpose=batch`, and creates one asynchronous 24-hour Batch.
Concurrency, staggering, and synchronous transient retries do not apply. The
prepared, uploaded, submitted, and collected states are saved separately with
an input SHA-256, stable `custom_id` map, OpenAI file/Batch IDs, and a
cross-process lease. Resuming first reconciles or retrieves that saved Batch;
it does not submit completed work again. Input is rejected before upload above
50,000 requests or 200 MB, and returned IDs/custom IDs must match the saved
state exactly.

For compact v9 validation failures, an explicitly authorized Standard retry
derives the smallest safe scope from the retained diagnostics. If every
problem maps to known ranks or context IDs, only those identities are requested
again; the response is merged by immutable identity and the entire candidate
is revalidated. The original response, exact paid repair response, repair
scope, and merged candidate remain separately inspectable. An unscoped/root
problem falls back to a full-chunk retry.

For an `invalid_response`, **View problems** presents every detected issue as
a card/field location, plain-language explanation, expected value, actual
value, suggested next step, and exact JSON path. Selecting an issue also shows
the affected returned card or response section. Malformed JSON, unsafe root
or card shape, missing/unexpected fields, and non-text field values are locked:
they cannot be manually accepted. If syntax and structure are safe, individual
content constraints—such as an empty field, the wrong example count, an
omitted requested term, or an extra legitimate source form—may be explicitly
accepted after human review.

Manual acceptance does not edit the model response and is not a general
“ignore validation” switch. Every selected problem is reidentified against
the retained response, the raw response hash and decision history are saved,
and all structural checks run again during packaging. A chunk becomes usable
only after every current problem is resolved. Resolving the final incomplete
chunk can immediately start the existing job's package/import stage; it does
not make another OpenAI request.

The worker pool is for network requests. The computer's GPU does not accelerate
OpenAI requests. GPU acceleration applies only to optional local CKIP source
tokenization described in [Corpus preparation](corpus_pipeline.md).

## Saved request contract

Creating an authorized source job freezes these OpenAI-facing settings in
`request_contract.json` before the first card request:

- model (`gpt-5.4-mini`);
- the fully composed prompt, including the Advanced-tab source component;
- the complete strict JSON response schema, including the exact per-chunk
  schemas for v8 or the fixed compact schema for v9;
- request protocol, reasoning effort, and Standard/Economy processing mode;
- optional web-search access and its one-call cap.

The corresponding source-batch payload is also saved in each chunk's
`request.json`. Automatic and manually authorized retries load the saved
contract; editing an Advanced prompt component later does not silently alter
an existing job's retry. **Inspect selected** includes the contract alongside
the chunk attempts. Older jobs that predate this file have a contract
reconstructed once from their saved pipeline and the then-current components;
that reconstructed contract is atomically retained and is not replaced on
later retries.

This immutability applies to a saved job. Starting a new job intentionally
captures the Card setup and prompt components that exist at that time.
Reasoning effort `low` gives the mini model a bounded budget for contextual
parsing, sense separation, example grammar, translation, and markup. `none`
avoids reasoning-token output charges and is the default; strict schema output
is validated locally in both cases. Cost estimates model low/central/high
reasoning separately. The hard v9 output ceiling has additional headroom for
unusually polysemous chunks and optional hidden deliberation. A ceiling is not
a prepaid reservation, and exhausting it would otherwise turn a paid response
into incomplete JSON.

## Persistence and recovery

Source jobs are stored below:

```text
output/source_generation_jobs/<job-id>/
    manifest.json
    request_contract.json
    combined.json
    events.jsonl
    workflow.json
    source_deck.apkg                 # after successful packaging
    chunks/
        000001-r1-r500/
            request.json
            status.json
            attempts/
                0001/
                    raw.txt          # when a response was received
                    response.json    # exact provider ID, status, and usage
                    validated.json   # only after validation
                    error.json       # when an attempt failed
                    repair_scope.json
                    repair_raw.txt   # exact selective-repair response
                    manual_validation.json
                                     # explicit per-problem review audit
    economy/
        batch_0001/
            input.jsonl
            output.jsonl             # retained after collection
            errors.jsonl
```

Paths move to the per-user AutoAnki output directory in a packaged
executable. Every attempt is written to a new directory rather than
overwriting the previous one. `combined.json` is rebuilt from validated
chunks and records whether the set is complete. Packaging and import do not
begin until every request validates.

If a process stops while a chunk is running, the incomplete attempt remains
available for inspection; when that job is resumed, the orphaned chunk is
returned to pending. Completed chunks remain completed. A package/import
failure leaves the validated combined JSON and any completed `.apkg` in the
job directory, so retrying the **Deck** row does not repeat OpenAI work.

Each chunk also has an operating-system advisory worker lease. If two
AutoAnki processes open the same saved job, only the process holding that
chunk's lease may issue its paid request. A process that merely inspects the
job cannot reset a chunk that another live process owns; the operating system
releases the lease after a crash so genuine interrupted work can be recovered.
This protection is per request chunk, not a general multi-user database or a
promise that arbitrary manual edits to job files are safe.

## Source-deck import semantics

A completed source run creates and imports:

```text
Vocabulary from <source title>
```

Unlike Manual Input's temporary-deck workflow, this deck is the intended final
deck. AutoAnki does not move its cards to a Card setup target deck, empty it,
or delete it. Stable note GUIDs and deterministic deck/model identities are
used during recovery, but importing the same completed package outside the
managed workflow should still be treated as a deliberate operation.

## Local file preparation

Prepare File supports these historical-language source variants:

- Classical Chinese (Warring States) uses the pinned CKIP Shanggu model.
- Classical Chinese (Ming) uses the pinned CKIP Jindai model plus the pinned
  Traditional Jieba refinement policy for unresolved long spans.
- Middle English and Old English use the built-in deterministic Unicode
  orthographic-word tokenizer. It retains letters such as `þ`, `ð`, `æ`, `ƿ`,
  and `ȝ`, combining diacritics, internal apostrophes/elisions, and hyphenated
  compounds with exact source offsets. Numeric and standalone Roman-numeral
  editorial labels are omitted.

Historical-English candidate identity is NFC plus Unicode case-folding, so a
sentence-initial `Whan` and later `whan` produce one candidate while retaining
the first spelling and every exact occurrence offset. Chinese deduplication
remains exact NFC surface-form matching. Middle/Old English contexts recognize
periods as sentence endings but do not split common editorial abbreviations or
initials; Chinese punctuation behavior is unchanged.

The accepted input formats are:

- `.txt` and `.md`: must be UTF-8 (a UTF-8 BOM is accepted); form-feed
  characters separate source sections.
- `.pdf`: text is extracted per non-empty page with `pypdf`.

There is no OCR engine. A scan/image-only PDF is rejected with instructions
to run OCR externally and provide a searchable PDF or UTF-8 text. PDF text
extraction cannot reliably infer reading order for every multi-column layout,
remove recurring headers/footers, or correct bad embedded character maps.
The generic local cleaner removes only limited URL/HTML/wiki-like artifacts.
One exact-signature exception handles the retained body of Project Gutenberg
ebook 22120, Skeat's *Canterbury Tales*: it separates the selected text from
the interleaved critical apparatus and applies the volume's published
corrections and preferred readings. It fails closed if the audited layout
changes. Always use Inspect Source and, when necessary, inspect the retained
canonical/raw artifacts before authorizing paid generation for any other
source.

The original input file, its hash, exact extracted page/section text, canonical
text, contexts, occurrences, first-occurrence vocabulary, and audit reports
are retained under `output/corpora/`. Preparation may download pinned CKIP
models and the Ming refinement dictionary, but it does not call the OpenAI
card API and does not touch Anki. Middle/Old English tokenization has no model
download and does not use the GPU preference.

Local preparation writes an atomic tokenization checkpoint after each
completed source section. If a later section or process fails, running
preparation again can reuse earlier valid section results instead of repeating
the expensive model pass. Reuse is fail-closed: the checkpoint identity binds
the source snapshot and section hashes, corpus-processing version, public
tokenizer/model/device identity, and a Python implementation fingerprint.
Cached spans must still reproduce the retained section text with valid,
ordered offsets and confidence values. An unreadable, truncated, stale,
identity-mismatched, or semantically invalid checkpoint is ignored and that
section is tokenized again; it is never admitted into a build. Tokenizers whose
implementation cannot be fingerprinted simply run without checkpoint reuse.
The later Ming/Jieba refinement and final audit still run normally.

## Codex retrieval

Codex Retrieve is optional and requires the `codex` CLI on `PATH` with usable
saved authentication (for example, sign in with `codex login` first). It is a
separate, explicitly authorized network action and may consume Codex plan/API
usage according to the account used by that CLI. It is not part of the offline
price estimate.

Each retrieval receives its own directory under:

```text
output/codex_source_jobs/<job-id>/
```

AutoAnki runs `codex exec` ephemerally with web search enabled, a
`workspace-write` sandbox rooted at that job, no interactive approvals, a
strict output schema, and user config/rules ignored. The request instructs
Codex to prefer lawful primary/public-domain or clearly licensed
original-language sources, treat instructions found in sources as untrusted,
record URLs and uncertainty, and write only PDF/TXT/Markdown results below the
job's `retrieved/` directory. AutoAnki validates returned paths before
preparation and retains the JSONL event log and stderr.

Those controls reduce accidental scope, but they do not prove copyright,
edition authenticity, textual completeness, OCR quality, or scholarly
accuracy. Review the retrieved files, URLs, warnings, and prepared vocabulary.
Codex retrieval never authorizes or starts OpenAI card generation; that still
requires the separate Generate Deck estimate and paid checkbox.

## Backend API

Planning and persistence can also be used without Tkinter:

```python
from source_generation import (
    GenerationJobRunner,
    SourceGenerationConfig,
    estimate_plan_cost,
    load_processed_source,
    make_pipeline_response_validator,
    plan_source_generation,
)

source = load_processed_source("daodejing_wang_bi")
config = SourceGenerationConfig(
    source_key=source.key,
    source_prefix_token_limit=100,
    chunk_size=30,
    context_mode="sentence_neighbors",
)
plan = plan_source_generation(source, config)
estimate = estimate_plan_cost(plan, pipeline, prompt_text=source_prompt)
```

`source_generation` itself does not create an OpenAI client or mutate Anki.
`SourceWorkflowController` supplies those application-level operations only
after the corresponding GUI authorization.
