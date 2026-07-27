# AutoAnki implementation tracker — 2026-07-26

This checklist preserves the acceptance criteria for the current multi-part
request. An item is complete only when its implementation and regression test
have both been verified.

## Prompt and source-card behaviour

- [x] Additional senses are limited to genuinely disjoint, common meanings a
  learner is reasonably likely to encounter. The current prompt already says
  this strongly, so its wording must not be changed without user approval.
- [x] Active and compatibility prompts request exactly three generated
  examples and aligned translations, with no model-side emboldening or HTML.
- [x] When local processing emphasizes the term in an example, do not display
  a duplicate term label. If emphasis fails, display the unbolded term below
  the sentence.
- [x] Generate-from-source exposes independent Sentence → Meaning,
  Word → Meaning, and Meaning → Word selections.
- [x] Retained-source Sentence → Meaning uses one note per original sentence:
  Original Sentence (front), English Translation (back), and optional Nuance
  (back).
- [x] Original source sentences are inserted locally rather than echoed by the
  model.
- [x] Generate-from-source exposes an optional sentence Nuance checkbox.
- [x] Source word definitions are requested only for selected lexical card
  directions.
- [x] Generate-from-source can package selected directions together or in
  separate decks.
- [x] Combined generated-example ordering is source-word order, then
  Sentence → Meaning, Word → Meaning, Meaning → Word.
- [x] Combined retained-source ordering is source-word order, with
  Word → Meaning before Meaning → Word, and one source sentence only after all
  associated selected words (including associations to excluded words).
- [x] Contextual lexical senses carry no duplicated source example; additional
  senses carry three generated examples and three aligned translations.

## Repository and application behaviour

- [x] Audit tracked files and `.gitignore`; untrack only confirmed generated,
  cache, runtime, recovery, or local-environment artifacts.
- [x] Profile GUI launch and make startup as close to instant as practical
  without changing user-visible behavior.
- [x] Persist ordinary menu/control settings across restarts.
- [x] Keep paid-action authorizations and advanced prompt-edit safeguards
  ephemeral.
- [x] Add an emergency pause that prevents queued/unsent OpenAI dispatches.
- [x] Add durable resume/recovery with visible Connecting, Successfully
  resumed, Connection timeout, No connection, and appropriate failure states.
- [x] Triple the useful Jobs list and inspection-reader space.
- [x] Make Jobs & Failures vertically scrollable using the established smooth
  rounded-scrollbar behavior.
- [x] Prevent mouse-wheel events over every dropdown from changing selection.
- [x] Put Meaning → Word pronunciation on the revealed word side, never the
  meaning side.

## Enhanced cards and local audio

- [x] Add an opt-in Enhanced card category.
- [x] Generate and package word/sentence audio as Anki media.
- [x] Use Style-Bert-VITS2 JP-Extra with an installed neutral Japanese voice.
- [x] Use Fun-CosyVoice 3 0.5B for English, French, and Chinese.
- [x] Store models in a shared user cache usable by other repositories.
- [x] Use GPU acceleration when supported, with a clear unavailable-runtime
  error rather than silent paid or CPU fallback.
- [x] For Sentence → Meaning and Word → Meaning, choose a stable 50/50
  audio-first or text-first presentation per card.
- [x] Audio-first fronts autoplay and expose a replay button; written content
  appears first on the back.
- [x] Text-first fronts put autoplay/replay audio first on the back.
- [x] Preserve ordinary cards and existing saved jobs when Enhanced cards are
  disabled.

## Final verification

- [x] Full automated regression suite (623 tests passed).
- [x] GUI startup/profile comparison and launch smoke test.
- [x] Settings restart round trip.
- [x] Pause/resume race and restart recovery tests.
- [x] Same-deck and split-deck ordering/package inspection.
- [x] Enhanced-card APKG media/template inspection and local synthesis smoke
  tests for every supported language family.
