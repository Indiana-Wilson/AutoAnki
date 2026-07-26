# Compact v10 evaluation report

Date: 2026-07-26

Source: Daodejing (Wang Bi)

Model: `gpt-5.4-mini`

## Outcome

Compact v10 removes provider-authored emphasis from both the prompt contract
and response schema. AutoAnki now strips safe presentation markup, restores
immutable terms and source examples locally, adds exact-match emphasis when
possible, and displays the term above every context-card example as the
fallback.

The first 200 unique source words completed at the recommended tested ceiling
of 30 words per request. The final result contains 384 validated cards:

| Ranks | Initial result | Final resolution | Cards | Standard cost |
|---|---:|---|---:|---:|
| 1–30 | Valid | None required | 61 | US$0.03744 |
| 31–60 | One bad rank | Selectively replaced rank 41 only | 59 | US$0.04237 including retry |
| 61–90 | Two unusable optional senses | Two audited local deletions | 58 | US$0.03917 |
| 91–100 | Valid | None required | 14 | US$0.01396 |
| 101–130 | Valid | None required | 55 | US$0.03658 |
| 131–160 | Valid | None required | 52 | US$0.03667 |
| 161–190 | Valid | None required | 70 | US$0.04225 |
| 191–200 | Valid | None required | 15 | US$0.01381 |

The production-shaped 200-word run cost US$0.26225 (about A$0.37598 at the
retained rate). Economy Batch would halve the token portion of the same frozen
requests. All exploratory calls, including deliberately retained failed
stress tests, cost US$0.33122 in total.

## Batch-size findings

- No reasoning passed at 1, 3, and 10 words with no local repairs.
- No reasoning at 30 words returned only one of 30 ranks despite a 71,064-token
  output ceiling. The API marked the response complete, so this was not
  truncation.
- Low reasoning at 30 words returned complete coverage.
- A single 60-word low-reasoning request passed structural validation but
  returned only 10 additional senses, versus 31 for the shared first 30 ranks
  in the 30-word request. This is a silent quality loss.

The safe default is therefore low reasoning with 30-word chunks. No reasoning
remains available for smaller experimental chunks. Requests larger than 30
were not used to claim the 200-word success.

## Real repair cases

- Rank 41 (`已`) had a valid optional “cease” sense, but three examples used
  the different character `以`. AutoAnki did not guess replacements. A
  one-rank retry was merged into the other 29 retained results.
- Rank 71 (`唯`) was given a `為` sense and none of its examples contained
  `唯`. The whole optional object was safely discarded.
- Rank 84 (`民心`) was given examples only for the component `心`. The whole
  component-only optional object was safely discarded.

The zero-occurrence deletion is Classical-Chinese-specific and requires four
well-formed aligned examples. If even one example contains the complete term,
the sense is retained and the bad examples remain eligible for selective
retry. Inflecting languages are exempt.

## Retained failures

The original two-chunk, 58-rank failed job now validates under v10 without
another provider call:

- ranks 1–30: 64 cards, 136 safe markup removals, zero remaining problems;
- ranks 31–58: 51 cards, 92 safe markup removals, 10 component-only optional
  senses discarded, zero remaining problems.

The earlier 100-word v9 low- and no-reasoning responses also both validate
under v10 after local markup removal.

## Cost changes

V10 keeps the same model. It removes all Wang-Bi-specific payload hints:

- 58 ranks: 17 hints and 6,038 payload characters removed;
- 100 ranks: 24 hints and 8,731 payload characters removed.

For the retained 58-rank shape, estimated fixed input falls from 9,482 tokens
under v9 to 8,047 under v10. At a 30-word request, prompt + payload + schema
falls from about 5,226 estimated tokens to about 3,986. The v10 prompt itself
is slightly longer because it carries general cross-language validity rules;
the payload reduction is much larger.

Measured v9 markup alone added about 247 estimated tokens to the old 10-word
response (52 strong spans). Those tags are now generated locally at no API
cost.

Output remains the dominant cost. Economy Batch is the largest direct price
reduction; prompt caching was observed on subsequent requests, with 1,792
input tokens billed at the cached rate.

## Repair boundary

A static inventory now has 60 reachable compact-protocol diagnostic
categories. At least 52 (86.7%) have an always-safe or predicate-gated local
repair path. Eight categories (13.3%) inherently require missing or corrected
semantic/identity content and are never guessed locally.

Raw provider output is never overwritten. Changed candidates are saved as
`repaired_raw.txt`, with SHA-256-linked rule/path/action records in
`local_repair.json`. Compact v9 and grouped v8 remain frozen rollback options.

## Luna compatibility evaluation

`gpt-5.6-luna` uses the same compact v10 contract, strict schema, validation,
local emphasis, Batch body, and repair protocol. It remains an experimental
selector rather than the default because the live Daodejing quality gates were
mixed:

| Words | Reasoning | Valid | Result | Standard cost |
|---:|---|---:|---|---:|
| 1 | none | Yes | 2 cards | US$0.00442 |
| 3 | none | Yes | 9 cards | US$0.01141 |
| 10 | none | Yes | 21 cards | US$0.01966 |
| 30 | low | No | 16 duplicate ranks plus non-ascending results | US$0.05692 |
| 100 | none | No | Empty arrays | US$0.00918 |
| 100 | low | Structurally yes | All 100 ranks, but zero additional senses | US$0.06156 |

The first one-word call also revealed that implicit Luna caching writes input
even without a cache key. Switching one-request jobs to explicit mode with no
breakpoint reduced cache writes from 2,042 to zero. Multi-request jobs use one
explicit breakpoint after the stable prompt and account separately for the
write and later reads.

## Approved follow-up changes

Implemented behind **Automatic repair**:

1. exact validated source-context translation memory, with conflicts disabled
   rather than guessed;
2. tiny strict-schema additional-sense example-pair repair;
3. at most three persisted automatic paid repair dispatches per failed chunk,
   with no automatic rank or whole-chunk fallback.

Corpus retrieval of additional-sense examples was declined. Modern-language
pronunciation/POS resources have a rollbackable v11 design in
`docs/language_resources_plan.md`; they remain unimplemented pending that
larger, separately reviewed resource milestone.
