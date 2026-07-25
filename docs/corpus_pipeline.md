# Corpus preparation

AutoAnki now contains a reusable, UI-ready corpus preparation subsystem under
`src/corpus_pipeline/`. It is intentionally isolated from `process_text.py`
and `anki_integration.py`: fetching, cleaning, tokenizing, auditing, and
writing inspection chunks never call OpenAI and never modify Anki.

The first two fixed source specifications are:

- **Daodejing:** Chinese Wikisource
  [老子（匯校版）](https://zh.wikisource.org/wiki/老子_(匯校版)), split into
  exactly 81 chapters.
- **Journey to the West:** the Chinese Wikisource
  [西遊記 index](https://zh.wikisource.org/wiki/西遊記) and exactly
  `西遊記/第001回` through `西遊記/第100回`.

These are Traditional Chinese historical texts, not modern translations or
simplified rewrites. Their provenance still needs to be described precisely:
`老子（匯校版）` is a Wikisource collation rather than a diplomatic
transcription of one manuscript, while the Wikisource Journey transcription
does not identify itself as a facsimile of one particular Ming printing. The
Daodejing source metadata currently labels its transcription quality at 50%;
the selected source is therefore reproducible and complete as a Wikisource
edition, but it is not represented here as an independently verified critical
text.

## Safe source extraction

The fetcher uses MediaWiki's revisions API in batches of at most 50, with a
descriptive User-Agent and `maxlag`. Every snapshot records the exact page
title, source URL, revision ID, revision timestamp, MediaWiki SHA-1, local
SHA-256, and untouched raw wikitext.

Raw wikitext is essential. Wikisource's rendered/plain-text extraction omits
a complete seven-line poem inside `<onlyinclude><poem>` in Journey chapter 90.
The strict cleaner instead:

- preserves prose, verse, punctuation, and original Traditional characters;
- removes site headers, footers, navigation, categories, structural headings,
  indentation markup, and Journey's exact closing `《西遊記》至此終。`
  colophon;
- retains the visible first argument of the documented `参`, `參`, and `另`
  textual-variant templates and discards editorial tooltips/alternates;
- retains the contents of the known chapter-90 poem containers;
- resolves the one known language-conversion span to `zh-hant`;
- keeps visible text from the one known interwiki literary-form link; and
- stops with an audit failure on any unknown content-bearing template, tag,
  link, missing page, reordered chapter, or leftover wiki/web markup.

The untouched raw revisions are also checked for isolated orthographic
variants inside the otherwise Traditional transcription. Cleaner version 4
normalizes only manually audited phrases containing Daodejing `还→還`,
`台→臺` and Journey `来→來`, `驼→駝`, `刬→剗`, `膻→羶`, `腭→齶`,
`晒→曬`. Some of those source forms have historical or variant uses, so this
is accurately described as source-specific orthographic normalization—not a
claim that every occurrence of a character is universally simplified.
Character-wide or broad OpenCC conversion is deliberately not used because
it could corrupt legitimate historical forms and meanings such as `后`, `云`,
`里`, `凶`, `采`, `几`, `帘`, `价`, `台`, and `膻`. The raw `.wiki` files
remain unchanged UTF-8 source records with fixed LF line endings.

Journey's literary chapter titles and Daodejing's numbered chapter labels are
retained as section metadata. Title inclusion is a build option, so either
choice reuses the same raw snapshot. Journey's meaningful literary titles are
included by default; Daodejing's bare numbered labels are excluded by default
because they are not chapter names. The CLI accepts `--section-titles` and
`--no-section-titles` to override either default. The From Source generation
UI consumes whichever audited build is currently published; title inclusion
for these presets remains a corpus-build/CLI choice rather than a paid
generation choice. Daodejing's `道經`/`德經` division headings remain
structural metadata and are excluded.

By default, a valid saved snapshot is reused. Passing `--refresh` explicitly
fetches current revisions and creates a new content-addressed snapshot.

## Historical word segmentation

The primary segmenters are CKIP Lab's period-specific Han Transformer models:

- `ckiplab/bert-base-han-chinese-ws-shanggu` for the Daodejing;
- `ckiplab/bert-base-han-chinese-ws-jindai` for Journey to the West.

The model revisions, B/I decoder version, and corpus-processing version are
hard-coded into build identity, so a later model or implementation update
cannot silently reuse an incompatible vocabulary run. Exact cleaned substrings
and offsets are retained; AutoAnki never converts the source through OpenCC.
Deduplication uses exact NFC-normalized Traditional surface form, and each word
is ranked by its first occurrence. Orthographic variants consequently remain
distinct—for example, `為` and `爲` are not silently merged.

Locally supplied Middle English and Old English files use a separate,
dependency-free Unicode orthographic tokenizer. A candidate begins with a
Unicode letter, retains combining diacritics and historical letters such as
thorn, eth, ash, wynn, and yogh, and may retain an internal apostrophe or true
hyphen joining two letter runs. Pure numbers and standalone editorial
Roman-numeral/page labels are not candidates. Every retained word span must be
covered exactly and reproduce the canonical source offsets or the audit fails.
Candidate identity uses NFC plus Unicode case-folding, so capitalization alone
does not create a second type; the first observed surface spelling remains on
the card. Period-aware sentence contexts are enabled only for the two
historical-English source keys and protect common abbreviations and initials,
leaving the established Chinese sentence boundaries unchanged.

The implementation was exercised on complete primary-work body slices from
two Project Gutenberg UTF-8 editions on 25 July 2026:

- Chaucer's *Canterbury Tales*,
  [ebook 22120](https://www.gutenberg.org/ebooks/22120) (Skeat edition),
  raw-file SHA-256
  `6dc8d8ae4b7cc4783bb0f6fb4b8b077f2768cdf00fd00c799f17722b70d928ff`.
  This body file interleaves Skeat's selected text with page and line
  numbers, headings, manuscript variants, critical notes, and rejected
  passages. Its exact-signature cleaner retains the selected work, resolves
  bracketed supplied readings, applies the volume's published corrections
  and preferred readings, and removes that presentation apparatus. The
  resulting canonical text has
  SHA-256
  `d2b791776b8d8a4922e8ef2a017d06e5acb0601a376e759b140a50a3cc41d434`
  and produced 181,270 exact occurrences, 11,914 case-folded candidates,
  and 7,071 sentence contexts.
- Harrison and Sharp's Old English *Beowulf*,
  [ebook 9701](https://www.gutenberg.org/ebooks/9701), raw-file SHA-256
  `5190b80b0829da81b1b375a5494909dc751aa365de6a88a816b576c048c54839`.
  The slice from `BĒOWULF. / I. THE PASSING OF SCYLD.` to the Finnsburg
  appendix produced 17,603 exact occurrences, 5,673 candidates, and 885
  sentence contexts.

Both complete builds passed `audit_build`, including exact substring offsets,
non-overlap, coverage of every orthographic word span, first-occurrence
ranking, contexts, and chunks. The Canterbury extractor is deliberately
limited to the audited ebook-22120 body signature and fails closed if its
layout or primary-text boundaries change. Generic Prepare File processing
cannot reliably decide which scholarly apparatus belongs to the primary text,
so every other user-supplied edition should still be trimmed or sectioned
deliberately before paid generation.

CKIP runs can use CPU, NVIDIA CUDA, Intel XPU, or Apple MPS when the installed
PyTorch build exposes that device. The default `auto` policy prefers CUDA,
then XPU, then MPS, and otherwise uses CPU. The resolved device, PyTorch and
Rust-tokenizers versions, Python version, operating system, and machine
architecture are recorded in the build identity. Shared packages are pinned
in `requirements-corpus-base.txt`; `requirements-corpus.txt` and
`requirements-corpus-gpu.txt` select the CPU or NVIDIA PyTorch wheel. This
prevents independently generated artifacts from appearing interchangeable
when their numerical runtime differs.

The historical models proved sensitive to unrelated sentences being combined
in one inference window. AutoAnki therefore sends each sentence (or
unterminated paragraph) to CKIP independently. Any all-Han result longer than
four characters is then presented to the same pinned model in isolation and
replaced only when that model finds a complete, offset-preserving partition.
Both behaviours have explicit version identifiers in the build manifest.

Journey still had 515 implausible long types after that same-model refinement.
For that work only, the default pipeline splits the remaining longer spans
with Jieba 0.42.1, `HMM=False`, and its Traditional Chinese large dictionary
pinned to repository revision
`67fa2e36e72f69d9134b8a1037b83fbb070b9775` and SHA-256
`b16011275c42955ccd81fc1adecc93a59dbb7926af69d93fc95d4943d40f6aad`.
The verified dictionary is cached under `output/corpora/_resources/`; its URL,
revision, hash, byte count, license, package version, and options are recorded
in manifests. If Jieba still returns a span over four Han graphemes, that span
is conservatively decomposed into individual graphemes. This avoids passing
whole clauses to paid generation, but it can fragment legitimate long names
and formulae. The Daodejing does not use this dictionary fallback.

CKIP segmentation is a statistical scholarly model, not a dictionary oracle.
Jieba is also a general dictionary segmenter, not a Ming-literature authority.
Either can still merge or split strings differently from a lexicographer, and
a four-character maximum is not proof that a result is one lexical word.
The generated list is therefore an ordered vocabulary **candidate** set. It
must be inspected before paid definition generation; it is not suitable for
unattended generation of every candidate. The retained contexts and offsets
make a later dictionary/manual review layer possible without re-scraping.
Structural completeness is stricter: mixed punctuation/Han model spans are
separated safely, and a build fails unless every Han source code point is
covered by exactly one non-overlapping token occurrence.

The large tokenizer dependencies and model downloads are deliberately absent
from the ordinary AutoAnki installation and base Windows executable. For a
reproducible CPU installation, run:

```text
.venv/bin/python -m pip install -r requirements-corpus.txt
```

For a supported NVIDIA GPU, use the CUDA requirements instead:

```text
.venv/bin/python -m pip install -r requirements-corpus-gpu.txt
```

On Windows, use `.venv\Scripts\python` instead. The GPU file currently selects
the official PyTorch CUDA 13.0 wheel; it requires a compatible NVIDIA driver.
Check PyTorch's current platform selector at
<https://pytorch.org/get-started/locally/> when moving to another driver,
CUDA, Python, or PyTorch release. Verify the environment before a long run:

```text
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Selecting the GPU checkbox in AutoAnki requests the `auto` policy. If the
installed PyTorch build cannot expose CUDA, `auto` records and uses CPU rather
than pretending that the GPU was used. Untick the option to require CPU.
Installing a CPU-only PyTorch wheel cannot use an NVIDIA GPU even when the
computer has one.

Each selected model is roughly 400 MB and is downloaded to the normal Hugging
Face cache on first use. CKIP Han Transformers and its model cards declare
GPL-3.0; this optional runtime dependency must be considered before
redistributing a bundled commercial application. Jieba and the pinned
Traditional dictionary are MIT-licensed.

`build_windows.bat` installs only `requirements.txt`. Consequently,
`dist\AutoAnki.exe` does not contain the CKIP/Jieba/PyTorch stack, downloaded
model weights, or the large Daodejing/Journey reference artifacts. Extending
the PyInstaller specification and distributing/licensing those assets is a
separate packaging task. The base executable can still use Manual Input,
prepare Middle/Old English files with the built-in tokenizer, and generate
from compatible processed artifacts provisioned separately, but it cannot
prepare a new historical-Chinese file by itself.

There is no silent character-tokenizer or modern-Chinese fallback. If the
selected historical model, pinned dictionary, or exact required package
version is unavailable, the build stops and reports the failure.
`CorpusService.build()` follows each corpus catalogue's refinement policy when
its `refiner` argument is omitted; passing `refiner=None` explicitly disables
that policy. This distinction is retained for a future advanced UI control.

## Running and inspecting

From the repository root:

```text
.venv/bin/python src/corpus_cli.py list
.venv/bin/python src/corpus_cli.py fetch --work daodejing_wang_bi
.venv/bin/python src/corpus_cli.py build --work daodejing_wang_bi
.venv/bin/python src/corpus_cli.py fetch --work daodejing_mawangdui
.venv/bin/python src/corpus_cli.py build --work daodejing_mawangdui
.venv/bin/python src/corpus_cli.py fetch --work journey_to_the_west
.venv/bin/python src/corpus_cli.py build --work journey_to_the_west
```

`prepare` combines fetch/reuse and build. `audit --path PATH` revalidates a
saved snapshot or run. The CLI states explicitly that these commands do not
call OpenAI or Anki. Add `--section-titles` or `--no-section-titles` to
`build` or `prepare` to override the selected work's default.

The same processed builds appear under **Generate > From Source**. Use
**Inspect Source** to page through first-seen candidates and their retained
section/neighboring-sentence metadata. **Prepare File** accepts a local PDF,
UTF-8 TXT, or Markdown file and publishes it into this same catalogue.
That local-file path supports Warring States, early-Han Mawangdui,
Wang-Bi-recension, and Ming historical Chinese plus Middle and Old English.
The historical-English tokenizer ships with the base application and is
deterministic; the Chinese variants retain their optional pinned-model
requirements.

PDF extraction is text-only and page-aware; it does not perform OCR. A scanned
PDF must be OCRed externally first. Embedded PDF text can have incorrect
reading order, repeated headers/footers, or broken character maps, while the
local cleaner removes only limited URL/HTML/wiki-like artifacts. The untouched
input and extracted page text are retained so the result can be audited rather
than silently repaired.

For locally prepared files, CKIP tokenization is checkpointed atomically after
each source section. A retry after a late failure reuses only checkpoints whose
saved identity and token spans fully validate. The identity covers the source
snapshot and section hashes, corpus-processing version, public
tokenizer/model/device configuration, and a Python implementation fingerprint;
each span must reproduce the exact retained section text in order. Corrupt,
partial, stale, or identity-mismatched files are ignored and recomputed from
the retained text. An opaque tokenizer implementation disables reuse instead
of weakening that check. Refinement and the complete semantic audit are still
performed after checkpointed tokenization, so a checkpoint is an optimization,
not a published build or an audit bypass.

These local checkpoints live below the source's
`checkpoints/tokenization/<snapshot-id>/` tree. They do not make the large
tokenizer dependencies or model weights part of the base Windows executable,
and they do not make OpenAI or Anki calls.

Runs are content-addressed and published atomically under:

```text
output/corpora/<work-key>/
    snapshots/<snapshot-id>/
        manifest.json
        source/*.wiki
        clean/text.txt
        clean/sections.jsonl
    runs/<build-id>/
        manifest.json
        source/*.wiki
        clean/text.txt
        clean/sections.jsonl
        contexts.jsonl
        occurrences.jsonl
        unique_words.jsonl
        chunks/1-500.txt
        chunks/1-500.json
        chunks/501-1000.txt
        ...
        audit/report.json
        audit/report.txt
        audit/chapter_boundaries.tsv
```

Text chunks contain one unique word per line for easy manual inspection.
Matching JSON chunks include the first occurrence, exact source offsets,
chapter ID/title, occurrence count, paragraph context, and the preceding,
current, and succeeding sentence IDs and texts. A boolean identifies candidates
whose first occurrence is in a chapter title. The first/last sentence naturally
uses `null` for its missing neighbour. The chapter-boundary report makes it
practical to inspect every chapter's first/last text and character count.

All serialized relative paths use portable `/` separators. Reusing a
content-addressed snapshot or build first validates artifact hashes and
deserializes the complete saved build. The semantic audit then rechecks
context IDs, section membership, non-overlapping token offsets, source order,
complete Han-character coverage, first-occurrence fields, counts, ranks, and
the actual contents of every text/JSON chunk.

## Verified reference runs

The locally prepared reference runs were independently re-read from their
serialized artifacts and audited on 24 July 2026:

| Work | Source structure | Han characters | Token occurrences | Unique candidates | Chunks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Daodejing | 81 chapters | 5,265 | 4,747 | 922 | 2 |
| Journey to the West | 100 titled chapters | 589,246 | 460,652 | 24,224 | 49 |

Their content-addressed snapshot/build pairs are
`0a5b8d83c82b4527e89baf43` / `23e76cbed91d61f34183391c`
and `2c7edf3b1938c493e7159a9a` / `6384315322c9cc5c6b0f6521`,
respectively.

The Daodejing chunks end with `501-922`; Journey has 48 full 500-candidate
chunks followed by `24001-24224`. Journey chapter 90's seven-line poem is
present. Independent scans confirmed that all retained Han/variation code
points are covered exactly once, all first-use ranks and occurrence counts
match the source, all stored neighbouring-sentence IDs resolve to their
matching texts, the closing colophon is absent, and every saved artifact
matches its manifest hash.

Journey's final candidates are one to four Han graphemes long. This deliberately
removes obvious clause-sized false tokens, but inspection still finds awkward
fragments in some named entities and Buddhist formulae. The preceding
same-model-only run is retained alongside the refined run, so a future review
tool can compare boundaries rather than requiring a new model download.

## AutoAnki source generation

The From Source UI now supplies the definition phase while keeping it separate
from corpus preparation:

1. choose Daodejing, Journey to the West, or a locally prepared source;
2. inspect the ordered candidate vocabulary and retained contexts;
3. select no context, the current sentence, current plus neighbouring
   sentences, or the complete span covered by each generation chunk;
4. select compact v9 or rollback-compatible legacy v8, no/low reasoning, and
   Standard or Economy processing;
5. vary the words per request and see the cache-aware
   prompt/schema/context/card-detail cost estimate update;
6. optionally omit exact visible field values from the union of the source
   language's saved Anki deck/note-type/field combinations; and
7. explicitly authorize the estimated OpenAI requests.

Identical and overlapping sentence windows are shared within each request;
words point to a context record instead of duplicating its text. Standard mode
makes one OpenAI request per generation chunk through a bounded, staggered
worker pool. Economy mode submits the frozen request bodies as one recoverable
24-hour Batch at half the Standard token rates. Paid responses and exact
provider-reported token usage are retained before local validation, completed
chunks are not repeated, and invalid model output requires a manually
authorized retry. A Standard compact-v9 retry can selectively regenerate
safely identified failed ranks/contexts; otherwise it falls back to the
complete chunk. A fully validated run is imported as
`Vocabulary from <source>` and that final deck is not moved, emptied, or
deleted.

See [From Source generation](source_generation.md) for the complete UI, cost,
rate-limit, recovery, Anki exclusion, local-file, and Codex retrieval
contract. Its saved job contract freezes the composed prompt, strict schema,
model, compact-v9/legacy-v8 protocol, no/low reasoning effort,
Standard/Economy processing mode, and optional web-search access so a retry
cannot change when an Advanced component is edited. Cost estimation, corpus
preparation, and the automated test suite do not call the OpenAI card API.
Codex Retrieve is a distinct, separately authorized network action, and card
generation has its own paid confirmation.
