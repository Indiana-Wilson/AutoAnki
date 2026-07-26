# Modern-language local resource plan

This is the approved design for moving pronunciation and unambiguous
part-of-speech work off the model. It is intentionally staged as a separate
implementation milestone: resource downloads and a partial lexical response
protocol are a larger change than the validation/repair work.

## User-visible behavior

The feature will live behind **Automatic repair**. Dictionary senses,
definitions, translations, register, nuance, and examples remain
model-authored.

- Pronunciation is supplied locally when an exact lookup collapses to one
  normalized reading. Missing or genuinely ambiguous readings are requested
  from the model.
- Part of speech is supplied locally only when every exact matching resource
  entry collapses to one broad part of speech. Ambiguous terms are resolved by
  the model for each model-authored sense.
- No resource definition is used as a dictionary sense.
- Historical and archaic languages are disabled unless a resource explicitly
  declares the exact AutoAnki language/period key.

## Rollbackable protocol

Implement this as a new v11 enrichment protocol. Leave v8-v10 request bytes,
schemas, fingerprints, and retries unchanged when Automatic repair is off.

1. Resolve exact local facts before the estimate and freeze resource versions,
   hashes, facts, misses, and ambiguities into the authorization.
2. The main v11 response omits pronunciation and POS fields selected for local
   enrichment. All dictionary senses still come from the model.
3. Inject singleton local values without altering provider `raw.txt`; retain a
   `lexical_enrichment.json` provenance sidecar and an effective merged body.
4. Send one tiny strict-schema fallback request only for unresolved
   `(rank, sense, field)` identities. It uses the model-created sense and
   context, and shares the existing maximum-three automatic-repair budget.
5. Run the ordinary complete card validator after the merge.

This avoids making every item echo local facts and avoids JSON Schema
branching for many per-word field masks.

## Initial resource adapters

Resource selection must use the exact `pipeline.language_key`, never the
broader `model_language_key`.

| Language key | Pronunciation | Part of speech | Initial status |
|---|---|---|---|
| `english` | Pinned [CMUdict](https://github.com/cmusphinx/cmudict), deterministic ARPAbet-to-General-American IPA; ambiguous variants fall back | Pinned [Open English WordNet](https://en-word.net/downloads); fill only one surviving n/v/adj/adv class | Planned |
| `french` | Pinned [Lexique](https://www.lexique.org/) exact NFC surface and unique `Phono_IPA` | Lexique category collapsed only when unique | Planned |
| `japanese` | Pinned [JMdict](https://www.jmdict.org/jmdict/j_jmdict.html) exact orthography with reading restrictions | Audited broad mapping of JMdict tags; multi-role entries fall back | Planned |
| Middle/Old English, Latin | None | None | Disabled pending exact-period adapters |
| Every Classical Chinese variant | None | None | Disabled; modern Mandarin resources are not compatible |

Data packs remain optional and separately versioned. Manifests retain source
URL/revision/date, archive and index SHA-256, parser version, license, and
attribution. CC BY-SA resources require an explicit redistribution review and
license notice before bundled distribution.

## Safe portable rules

- NFC exact-surface lookup; no fuzzy matching, case folding, accent stripping,
  lemmatization, or cross-period fallback.
- A local fact is used if and only if its normalized candidate set has exactly
  one value.
- Missing, corrupt, stale, or ambiguous data fails closed to the model.
- Facts and resource identities are frozen before a paid request.
- Resource changes revoke the estimate authorization.
- Output-language labels are filled locally only when an audited renderer
  exists; an English POS label is never inserted into a French/Japanese field.

Language-specific adapters still own dialect/IPA conversion, inflected-form
coverage, spelling-reading restrictions, and POS tag collapse.

## Acceptance gates

- Exact Unicode, apostrophe/hyphen, inflection, ambiguity, corrupt pack, and
  hash-mismatch tests for each adapter.
- Historical-language leak tests, especially Middle/Old English sharing the
  broad English model key and all Classical Chinese variants.
- All-local, mixed, and no-local v11 response/fallback tests.
- Interrupted-run recovery at main response, enrichment, and fallback stages
  without repeating a completed paid call.
- Cost estimates subtract main-response pronunciation/POS tokens and display
  exact local hit, miss, ambiguity, and bounded fallback costs.
- Frozen v8-v10 fingerprints remain byte-for-byte unchanged.
