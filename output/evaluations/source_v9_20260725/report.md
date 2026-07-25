# Source-generation cost overhaul

Date: 2026-07-25  
Corpus: Daodejing, received Wang Bi recension  
Estimate case: 944 candidate terms, 30 terms/request, sentence context, source
sentences reused as examples, English example/context translations, no web
search.

## Bottom line

The model did not become more expensive: the old and new paths both use
`gpt-5.4-mini`. The cost increase came from the request/estimate shape, not a
model upgrade.

Compact v9 reduces estimated full-corpus input from 1,385,797 to 135,353
tokens (90.2%). With no reasoning, its central Standard estimate is A$2.447
instead of legacy v8's A$4.158 (41.1% lower). Optional Economy Batch reduces
the v9/no-reasoning central estimate to A$1.224 (70.6% below legacy v8
Standard) without changing the model, prompt, response schema, or requested
cards.

The remaining projected cost is overwhelmingly response output, principally
the lexical fields and four generated sentence/translation pairs for every
additional sense. In the v9/no-reasoning central estimate, input is A$0.075
and output is A$2.372.

## Exact estimated cost components

| Input component | Legacy v8 | Compact v9 | Change |
|---|---:|---:|---:|
| Repeated prompt | 190,368 | 57,792 | -69.6% |
| Source payload | 101,535 | 60,601 | -40.3% |
| Response schema | 1,093,894 | 16,960 | -98.4% |
| Total input | 1,385,797 | 135,353 | -90.2% |
| Cache-eligible input in central case | 184,419 | 72,416 | — |

V8 repeated a 5,949-token prompt and a dynamic 16,331–35,285-token schema in
each of 32 requests. V9 uses a 1,806-token prompt and one fixed 530-token
schema. Dynamic rank/context identity remains in the compact payload and is
validated locally.

| Protocol | Reasoning | Execution | Low A$ | Central A$ | High A$ | Central change vs v8 |
|---|---|---|---:|---:|---:|---:|
| Legacy v8 | Low | Standard | 2.970 | 4.158 | 7.431 | baseline |
| Compact v9 | Low | Standard | 1.733 | 2.922 | 6.086 | -29.7% |
| Compact v9 | None | Standard | 1.654 | 2.447 | 4.106 | -41.1% |
| Compact v9 | Low | Economy | 0.867 | 1.461 | 3.043 | -64.9% |
| Compact v9 | None | Economy | 0.827 | 1.224 | 2.053 | -70.6% |

The range is an explicit response-shape estimate, not a quote. Its central
case assumes one contextual sense and one additional sense per term; the live
100-term calls returned only 0.13–0.25 additional senses per term, so this
central projection is deliberately conservative for this corpus.

## What is sent and returned

Each request sends:

1. A shared static prompt describing the configured card fields and quality
   rules.
2. Compact term records: immutable rank, exact term, context ID, bounded
   marked occurrence, and an optional short audited hint.
3. Each deduplicated source context once.
4. One fixed strict JSON schema.

Each response requests:

1. One lexical-only contextual sense per term. The original source sentence
   is inserted locally, so the model does not echo it.
2. One English translation per deduplicated source context, also reused
   locally.
3. Every genuinely disjoint common additional sense. Each additional sense
   still contains its lexical fields, four generated source-language
   examples, and four aligned English translations. This is now the largest
   variable cost.
4. Optional reasoning tokens when Low is selected; OpenAI bills them as
   output tokens.

No web search was enabled in these tests. When enabled, actual search calls
are separately retained and priced.

## Live staged tests

All prices below are provider-reported token usage converted at the frozen
AUD/USD rate. A repair row's price is included in the aggregate.

| Terms | Reasoning | First response | Final cards | Repair scope | Aggregate A$ |
|---:|---|---|---:|---|---:|
| 1 | None | valid | 4 | none | 0.00757 |
| 1 | Low | valid | 4 | none | 0.00713 |
| 3 | None | valid | 10 | none | 0.01296 |
| 3 | Low | valid | 10 | none | 0.01969 |
| 10 | None | one markup defect | 23 | rank 2 | 0.03153 |
| 10 | Low | one markup defect | 23 | rank 10 | 0.03076 |
| 100 | None | one markup defect | 125 | rank 13 | 0.10535 |
| 100 | Low | four markup defects | 113 | ranks 1, 3, 7 | 0.10051 |

The 100-term no-reasoning call cost A$0.09969 before repair; repairing only
rank 13 added about A$0.00565. The low-reasoning call cost A$0.08803 before
repair and A$0.10051 after repairing three ranks. Low did not show a dependable
quality advantage: it returned fewer additional senses and had several
contextual errors of its own. No reasoning therefore remains the default,
with Low available explicitly.

Manual inspection of the initial 100-term outputs found fine-grained
Classical Chinese errors that a generic structural validator cannot infer,
including the reading/sense of 見 in 不見可欲, discourse 夫, 智者, 民心, and
several earlier contrasts. Fourteen unequivocal Wang Bi occurrences now have
small term-local instructions plus immutable local checks for translation,
pronunciation, and grammatical role.

Post-change targeted retests:

| Audited ranks | Result | Cards | Aggregate A$ |
|---|---|---:|---:|
| 70, 83, 84, 95, 99 | valid first response | 9 | 0.01266 |
| 37, 38, 40, 65, 69 | valid after rank-38-only repair | 15 | 0.02378 |
| 11, 17, 21, 24 | valid after rank-17-only repair | 11 | 0.02029 |

The corrected outputs now retain, among other facts, 見 `xiàn` “display/show,”
夫 `fú` as a discourse particle, 智者 `zhì zhě`, the complete meaning of 民心,
徼 `jiào`, and contextual 辭 “decline/refuse.” This is a deterministic guard
for audited occurrences, not a claim that all 944 terms have received a full
scholarly lexical audit. The normal validation/review UI remains necessary for
unanchored semantic judgments.

## Reliability and rollback

- Compact v9 uses a fixed schema, immutable rank/context identities, exact
  array coverage checks, and local reconstruction of terms/source examples.
- A paid response and exact provider usage are atomically retained before
  validation. Recovery deduplicates repeated provider response IDs.
- Safe v9 failures are repaired by failed rank/context only; unscoped failures
  fall back to a full-chunk retry.
- The hard v9 output ceiling has extra unpaid headroom, preventing an unusual
  polysemous request from ending solely because the expected-output estimate
  was too small.
- Economy persists prepared, uploaded, submitted, and collected states,
  verifies hashes/IDs/custom IDs, and resumes without blindly submitting a
  second Batch.
- Legacy v8 remains an explicit GUI rollback option. Existing v8 request bytes
  and saved contracts are not migrated.
- Jobs freeze protocol, reasoning, execution mode, model, prompt/schema, search
  setting, and cost authorization, so a resumed job cannot silently switch
  systems.

OpenAI documents automatic prompt caching for repeated prefixes of at least
1,024 tokens, with static content first and dynamic content last:
<https://developers.openai.com/api/docs/guides/prompt-caching>.

Economy uses the Batch API's 50% token rates, separate rate limits, and
24-hour completion window:
<https://developers.openai.com/api/docs/guides/batch>.

The frozen token rates used by this report are from:
<https://developers.openai.com/api/docs/pricing>.

## Verification

- Full automated suite: 495 tests.
- Python compilation checks: passed.
- `git diff --check`: passed.
- Paid evaluation artifacts, exact raw responses, validation reports, usage,
  prices, and repaired merges are retained beside this report.
