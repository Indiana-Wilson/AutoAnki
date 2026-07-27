"""Stable genanki models for AutoAnki's configurable vocabulary cards."""

from dataclasses import dataclass

import genanki


ENGLISH_CONTEXT_MODEL_ID = 2092222676
ENGLISH_WORD_TO_MEANING_MODEL_ID = 1507373510
ENGLISH_MEANING_TO_WORD_MODEL_ID = 1553534904
CLASSICAL_CHINESE_CONTEXT_MODEL_ID = 1511279316
CLASSICAL_CHINESE_WORD_TO_MEANING_MODEL_ID = 1499168913
CLASSICAL_CHINESE_MEANING_TO_WORD_MODEL_ID = 1334631148
FRENCH_CONTEXT_MODEL_ID = 1989780161
FRENCH_WORD_TO_MEANING_MODEL_ID = 1716933979
FRENCH_MEANING_TO_WORD_MODEL_ID = 1534973221
JAPANESE_CONTEXT_MODEL_ID = 1312496295
JAPANESE_WORD_TO_MEANING_MODEL_ID = 2057405784
JAPANESE_MEANING_TO_WORD_MODEL_ID = 2085400325
LATIN_CONTEXT_MODEL_ID = 2006934887
LATIN_WORD_TO_MEANING_MODEL_ID = 1738304675
LATIN_MEANING_TO_WORD_MODEL_ID = 2021716093
SOURCE_SENTENCE_MODEL_ID = 1942026101
ENHANCED_SOURCE_SENTENCE_MODEL_ID = 1190000041

DECK_ID = 2059400110
DECK_NAME = "Generated English Words"

TERM_FIELD_NAMES = {
    "english": "Word",
    "classical_chinese": "Classical Chinese",
    "french": "French",
    "japanese": "Japanese",
    "latin": "Latin",
}
LANGUAGE_NAMES = {
    "english": "English",
    "classical_chinese": "Classical Chinese",
    "french": "French",
    "japanese": "Japanese",
    "latin": "Latin",
}
DIRECTION_NAMES = {
    "context": "Context Sentence to Meaning",
    "word_to_meaning": "Word to Meaning",
    "meaning_to_word": "Meaning to Word",
}
CONTENT_FIELDS = (
    ("translation", "Translation"),
    ("dictionary_meaning", "Dictionary Meaning"),
    ("pronunciation", "Pronunciation"),
    ("part_of_speech", "Part of Speech"),
    ("register", "Register"),
    ("nuance", "Nuance"),
)
CONTENT_FIELD_NAMES = tuple(
    field_name
    for _field_key, field_name in CONTENT_FIELDS)
SENTENCE_TRANSLATIONS_FIELD_NAME = "Sentence Translations (English)"
SOURCE_ORIGINAL_SENTENCE_FIELD_NAME = "Original Sentence"
SOURCE_ENGLISH_TRANSLATION_FIELD_NAME = "English Translation"
SOURCE_SENTENCE_NUANCE_FIELD_NAME = "Nuance"
SOURCE_CONTEXT_BLOCK_PREFIX = "\ue000S"
SOURCE_CONTEXT_ESCAPE_MARKER = "\ue000"
WORD_AUDIO_FIELD_NAME = "Word Audio"
ENHANCED_SENTENCE_FIELD_NAME = "Enhanced Sentence"
ENHANCED_SENTENCE_TRANSLATION_FIELD_NAME = (
    "Enhanced Sentence Translation")
SENTENCE_AUDIO_FIELD_NAME = "Sentence Audio"
AUDIO_FIRST_FIELD_NAME = "Audio First"
WRITTEN_FIRST_FIELD_NAME = "Written First"
PRESENTATION_KEY_FIELD_NAME = "Presentation Key"
ENHANCED_FIELD_NAMES = (
    WORD_AUDIO_FIELD_NAME,
    ENHANCED_SENTENCE_FIELD_NAME,
    ENHANCED_SENTENCE_TRANSLATION_FIELD_NAME,
    SENTENCE_AUDIO_FIELD_NAME,
    AUDIO_FIRST_FIELD_NAME,
    WRITTEN_FIRST_FIELD_NAME,
    PRESENTATION_KEY_FIELD_NAME,
)

CARD_CSS = """.card {
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  margin: 0;
  padding: 24px;
  font-size: 20px;
  text-align: center;
}

.definition-stack {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  gap: 0.85em;
  max-width: 44em;
  text-align: center;
}

.definition-section {
  line-height: 1.35;
  text-align: center;
}

.definition-label {
  display: block;
  margin-bottom: 0.15em;
  font-weight: 700;
}

.term {
  text-align: center;
}

.term-pronunciation {
  margin-top: 0.35em;
}

.enhanced-audio {
  min-height: 2.4em;
  display: flex;
  align-items: center;
  justify-content: center;
}

.enhanced-audio .replay-button,
.enhanced-audio .replaybutton,
.enhanced-audio .soundLink {
  cursor: pointer;
}

.context-example {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  gap: 0.65em;
  max-width: 44em;
}

.context-fallback-term {
  max-width: 44em;
  font-weight: 400;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.sentence {
  max-width: 44em;
  line-height: 1.65;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

.sentence-translation {
  max-width: 44em;
  margin-bottom: 0.9em;
  line-height: 1.5;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}
"""


def _definition_stack(
        *,
        excluded_field_keys=(),
        empty_fallback=None):
    excluded_field_keys = frozenset(excluded_field_keys)
    sections = []
    included_field_names = []
    for field_key, field_name in CONTENT_FIELDS:
        if field_key in excluded_field_keys:
            continue
        included_field_names.append(field_name)
        sections.append(
            f"""{{{{#{field_name}}}}}
<div class="definition-section">
  <strong class="definition-label">{field_name}</strong>
  <div>{{{{{field_name}}}}}</div>
</div>
{{{{/{field_name}}}}}""")
    if empty_fallback:
        opening_guards = "\n".join(
            f"{{{{^{field_name}}}}}"
            for field_name in included_field_names)
        closing_guards = "\n".join(
            f"{{{{/{field_name}}}}}"
            for field_name in reversed(included_field_names))
        sections.append(
            f"""{opening_guards}
<div class="definition-section">{empty_fallback}</div>
{closing_guards}""")
    return (
        '<div class="definition-stack">\n'
        + "\n".join(sections)
        + "\n</div>")


def _meaning_to_word_answer(term_field):
    return f"""<div>{{{{FrontSide}}}}<hr>
<div class="term">
  <div>{{{{{term_field}}}}}</div>
  {{{{#Pronunciation}}}}
  <div class="term-pronunciation">{{{{Pronunciation}}}}</div>
  {{{{/Pronunciation}}}}
</div>
</div>"""


def _autoplay_script():
    # This mirrors the user's established Anki template.  The replay control
    # remains visible; the click makes playback reliable on installations
    # where deck autoplay has been disabled.
    return """<script>
(function () {
  const root = document.getElementById("autoplay");
  if (!root) return;
  const button = root.querySelector(
    ".soundLink, .replaybutton, .replay-button");
  if (button) button.click();
})();
</script>"""


def _enhanced_audio_block(field_name):
    return (
        '<div id="autoplay" class="enhanced-audio">'
        f"{{{{{field_name}}}}}"
        "</div>"
        + _autoplay_script())


def _presentation_anchor():
    # genanki must be able to identify at least one unconditionally required
    # front field.  The stable key is populated on every Enhanced note and is
    # deliberately hidden from the learner.
    return (
        f'<span hidden aria-hidden="true">'
        f"{{{{{PRESENTATION_KEY_FIELD_NAME}}}}}"
        "</span>")


def _enhanced_sentence_block(term_field, identity):
    """Render one selected sentence and the optional plain fallback below it."""
    sentence_id = f"enhanced-sentence-{identity}"
    fallback_id = f"enhanced-fallback-{identity}"
    return f"""<div class="context-example">
  <div id="{sentence_id}" class="sentence">
    {{{{{ENHANCED_SENTENCE_FIELD_NAME}}}}}
  </div>
  <div id="{fallback_id}" class="context-fallback-term" hidden>
    {{{{{term_field}}}}}
  </div>
</div>
<script>
(function () {{
  const sentence = document.getElementById("{sentence_id}");
  const fallback = document.getElementById("{fallback_id}");
  if (
      sentence
      && fallback
      && fallback.textContent.trim()
      && !sentence.querySelector("strong")) {{
    fallback.hidden = false;
  }}
}})();
</script>"""


def _enhanced_context_question(term_field):
    return f"""{_presentation_anchor()}
{{{{#{AUDIO_FIRST_FIELD_NAME}}}}}
{_enhanced_audio_block(SENTENCE_AUDIO_FIELD_NAME)}
{{{{/{AUDIO_FIRST_FIELD_NAME}}}}}
{{{{#{WRITTEN_FIRST_FIELD_NAME}}}}}
{_enhanced_sentence_block(term_field, "front")}
{{{{/{WRITTEN_FIRST_FIELD_NAME}}}}}"""


def _enhanced_context_answer(term_field):
    return f"""<div>
{_enhanced_sentence_block(term_field, "back")}
{_enhanced_audio_block(SENTENCE_AUDIO_FIELD_NAME)}
<hr>
{{{{#{ENHANCED_SENTENCE_TRANSLATION_FIELD_NAME}}}}}
<div class="sentence-translation">
  {{{{{ENHANCED_SENTENCE_TRANSLATION_FIELD_NAME}}}}}
</div>
{{{{/{ENHANCED_SENTENCE_TRANSLATION_FIELD_NAME}}}}}
{_definition_stack()}
</div>"""


def _enhanced_question(language_key, direction_key):
    term_field = TERM_FIELD_NAMES[language_key]
    if direction_key == "context":
        return _enhanced_context_question(term_field)
    if direction_key == "word_to_meaning":
        return f"""{_presentation_anchor()}
{{{{#{AUDIO_FIRST_FIELD_NAME}}}}}
{_enhanced_audio_block(WORD_AUDIO_FIELD_NAME)}
{{{{/{AUDIO_FIRST_FIELD_NAME}}}}}
{{{{#{WRITTEN_FIRST_FIELD_NAME}}}}}
<div class="term">{{{{{term_field}}}}}</div>
{{{{/{WRITTEN_FIRST_FIELD_NAME}}}}}"""
    return (
        _presentation_anchor()
        + _definition_stack(
            excluded_field_keys=("pronunciation",),
            empty_fallback=(
                "No meaning field was configured for this card.")))


def _enhanced_answer(language_key, direction_key):
    term_field = TERM_FIELD_NAMES[language_key]
    if direction_key == "context":
        return _enhanced_context_answer(term_field)
    if direction_key == "word_to_meaning":
        return f"""<div>
<div class="term">{{{{{term_field}}}}}</div>
{_enhanced_audio_block(WORD_AUDIO_FIELD_NAME)}
<hr>
{_definition_stack()}
</div>"""
    return f"""<div>
{{{{FrontSide}}}}
<hr>
<div class="term">
  <div>{{{{{term_field}}}}}</div>
  {_enhanced_audio_block(WORD_AUDIO_FIELD_NAME)}
  {{{{#Pronunciation}}}}
  <div class="term-pronunciation">{{{{Pronunciation}}}}</div>
  {{{{/Pronunciation}}}}
</div>
</div>"""


def _context_front(term_field):
    return f"""<div id="sentence-source" hidden>{{{{Sentences}}}}</div>
<div class="context-example">
  <div id="sentence" class="sentence"></div>
  <div id="context-fallback-term" class="context-fallback-term" hidden>
    {{{{{term_field}}}}}
  </div>
</div>
<script>
(function () {{
  const source = document.getElementById("sentence-source");
  const raw = source ? source.innerHTML : "";
  const blockPrefix = "\\uE000S";
  const escapeMarker = "\\uE000";
  let sentences;
  if (raw.startsWith(blockPrefix)) {{
    const encoded = raw.slice(blockPrefix.length);
    sentences = [encoded
      .split(escapeMarker + "1").join("|")
      .split(escapeMarker + "0").join(escapeMarker)];
  }} else {{
    sentences = raw.includes("|")
      ? raw.split("|").map(s => s.trim())
      : (raw.trim() ? [raw.trim()] : []);
  }}
  const candidateIndices = sentences
    .map((sentence, index) => sentence ? index : -1)
    .filter(index => index >= 0);
  if (!candidateIndices.length) return;
  let hash = 2166136261;
  for (let index = 0; index < raw.length; index += 1) {{
    hash ^= raw.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }}
  const key = "autoanki-sentence-index-" + (hash >>> 0).toString(16);
  let chosenIndex = Number.parseInt(sessionStorage.getItem(key), 10);
  if (
      !Number.isInteger(chosenIndex)
      || !candidateIndices.includes(chosenIndex)) {{
    chosenIndex = candidateIndices[
      Math.floor(Math.random() * candidateIndices.length)];
    sessionStorage.setItem(key, String(chosenIndex));
  }}
  const sentenceElement = document.getElementById("sentence");
  sentenceElement.innerHTML = sentences[chosenIndex];
  const fallback = document.getElementById("context-fallback-term");
  if (
      fallback
      && fallback.textContent.trim()
      && !sentenceElement.querySelector("strong")) {{
    fallback.hidden = false;
  }}
}})();
</script>"""


def _context_back(term_field):
    return f"""<div>
{{{{FrontSide}}}}
<hr>
{{{{#{SENTENCE_TRANSLATIONS_FIELD_NAME}}}}}
<div id="sentence-translations-source" hidden>
  {{{{{SENTENCE_TRANSLATIONS_FIELD_NAME}}}}}
</div>
<div id="sentence-translation" class="sentence-translation"></div>
{{{{/{SENTENCE_TRANSLATIONS_FIELD_NAME}}}}}
{_definition_stack()}
</div>
<script>
(function () {{
  const sentenceSource = document.getElementById("sentence-source");
  const sentenceRaw = sentenceSource ? sentenceSource.innerHTML : "";
  let hash = 2166136261;
  for (let index = 0; index < sentenceRaw.length; index += 1) {{
    hash ^= sentenceRaw.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }}
  const key = "autoanki-sentence-index-" + (hash >>> 0).toString(16);
  const chosenIndex = Number.parseInt(sessionStorage.getItem(key), 10);
  const translationSource = document.getElementById(
    "sentence-translations-source");
  const translationRaw = translationSource
    ? translationSource.innerHTML
    : "";
  const translations = translationRaw.includes("|")
    ? translationRaw.split("|").map(s => s.trim())
    : (translationRaw.trim() ? [translationRaw.trim()] : []);
  const translation = (
      Number.isInteger(chosenIndex)
      && chosenIndex >= 0
      && chosenIndex < translations.length)
    ? translations[chosenIndex]
    : "";
  const target = document.getElementById("sentence-translation");
  if (target && translation) target.innerHTML = translation;
  sessionStorage.removeItem(key);
}})();
</script>"""


def _create_model(language_key, direction_key, model_id):
    language_name = LANGUAGE_NAMES[language_key]
    direction_name = DIRECTION_NAMES[direction_key]
    term_field = TERM_FIELD_NAMES[language_key]
    fields = [
        {"name": term_field},
        {"name": "Sentences"},
        *({"name": field_name} for field_name in CONTENT_FIELD_NAMES),
        # Append instead of inserting so every existing field keeps its Anki
        # ordinal when these stable model IDs are imported as an update.
        {"name": SENTENCE_TRANSLATIONS_FIELD_NAME},
    ]

    if direction_key == "context":
        question = _context_front(term_field)
        answer = _context_back(term_field)
    elif direction_key == "word_to_meaning":
        question = f'<div class="term">{{{{{term_field}}}}}</div>'
        answer = (
            "<div>{{FrontSide}}<hr>"
            + _definition_stack()
            + "</div>")
    else:
        question = _definition_stack(
            excluded_field_keys=("pronunciation",),
            empty_fallback=(
                "No meaning field was configured for this card."))
        answer = _meaning_to_word_answer(term_field)

    return genanki.Model(
        model_id,
        f"AutoAnki {language_name} - {direction_name}",
        fields=fields,
        templates=[{
            "name": f"{language_name} {direction_name}",
            "qfmt": question,
            "afmt": answer,
        }],
        css=CARD_CSS)


def _create_enhanced_model(language_key, direction_key, model_id):
    language_name = LANGUAGE_NAMES[language_key]
    direction_name = DIRECTION_NAMES[direction_key]
    term_field = TERM_FIELD_NAMES[language_key]
    return genanki.Model(
        model_id,
        f"AutoAnki Enhanced {language_name} - {direction_name}",
        fields=[
            {"name": term_field},
            {"name": "Sentences"},
            *({"name": field_name} for field_name in CONTENT_FIELD_NAMES),
            {"name": SENTENCE_TRANSLATIONS_FIELD_NAME},
            *({"name": field_name} for field_name in ENHANCED_FIELD_NAMES),
        ],
        templates=[{
            "name": f"Enhanced {language_name} {direction_name}",
            "qfmt": _enhanced_question(language_key, direction_key),
            "afmt": _enhanced_answer(language_key, direction_key),
        }],
        css=CARD_CSS)


def _create_source_sentence_model():
    return genanki.Model(
        SOURCE_SENTENCE_MODEL_ID,
        "AutoAnki Source Sentence - Sentence to Meaning",
        fields=[
            {"name": SOURCE_ORIGINAL_SENTENCE_FIELD_NAME},
            {"name": SOURCE_ENGLISH_TRANSLATION_FIELD_NAME},
            {"name": SOURCE_SENTENCE_NUANCE_FIELD_NAME},
        ],
        templates=[{
            "name": "Source Sentence to Meaning",
            "qfmt": (
                '<div class="sentence">{{Original Sentence}}</div>'),
            "afmt": (
                "<div>{{FrontSide}}<hr>"
                '<div class="sentence-translation">'
                "{{English Translation}}</div>"
                "{{#Nuance}}"
                '<div class="definition-section">'
                '<strong class="definition-label">Nuance</strong>'
                "<div>{{Nuance}}</div>"
                "</div>"
                "{{/Nuance}}"
                "</div>"),
        }],
        css=CARD_CSS)


def _create_enhanced_source_sentence_model():
    return genanki.Model(
        ENHANCED_SOURCE_SENTENCE_MODEL_ID,
        "AutoAnki Enhanced Source Sentence - Sentence to Meaning",
        fields=[
            {"name": SOURCE_ORIGINAL_SENTENCE_FIELD_NAME},
            {"name": SOURCE_ENGLISH_TRANSLATION_FIELD_NAME},
            {"name": SOURCE_SENTENCE_NUANCE_FIELD_NAME},
            {"name": SENTENCE_AUDIO_FIELD_NAME},
            {"name": AUDIO_FIRST_FIELD_NAME},
            {"name": WRITTEN_FIRST_FIELD_NAME},
            {"name": PRESENTATION_KEY_FIELD_NAME},
        ],
        templates=[{
            "name": "Enhanced Source Sentence to Meaning",
            "qfmt": f"""{_presentation_anchor()}
{{{{#{AUDIO_FIRST_FIELD_NAME}}}}}
{_enhanced_audio_block(SENTENCE_AUDIO_FIELD_NAME)}
{{{{/{AUDIO_FIRST_FIELD_NAME}}}}}
{{{{#{WRITTEN_FIRST_FIELD_NAME}}}}}
<div class="sentence">{{{{{SOURCE_ORIGINAL_SENTENCE_FIELD_NAME}}}}}</div>
{{{{/{WRITTEN_FIRST_FIELD_NAME}}}}}""",
            "afmt": f"""<div>
<div class="sentence">{{{{{SOURCE_ORIGINAL_SENTENCE_FIELD_NAME}}}}}</div>
{_enhanced_audio_block(SENTENCE_AUDIO_FIELD_NAME)}
<hr>
<div class="sentence-translation">
  {{{{{SOURCE_ENGLISH_TRANSLATION_FIELD_NAME}}}}}
</div>
{{{{#{SOURCE_SENTENCE_NUANCE_FIELD_NAME}}}}}
<div class="definition-section">
  <strong class="definition-label">Nuance</strong>
  <div>{{{{{SOURCE_SENTENCE_NUANCE_FIELD_NAME}}}}}</div>
</div>
{{{{/{SOURCE_SENTENCE_NUANCE_FIELD_NAME}}}}}
</div>""",
        }],
        css=CARD_CSS)


MODEL_IDS = {
    ("english", "context"): ENGLISH_CONTEXT_MODEL_ID,
    ("english", "word_to_meaning"): ENGLISH_WORD_TO_MEANING_MODEL_ID,
    ("english", "meaning_to_word"): ENGLISH_MEANING_TO_WORD_MODEL_ID,
    (
        "classical_chinese",
        "context",
    ): CLASSICAL_CHINESE_CONTEXT_MODEL_ID,
    (
        "classical_chinese",
        "word_to_meaning",
    ): CLASSICAL_CHINESE_WORD_TO_MEANING_MODEL_ID,
    (
        "classical_chinese",
        "meaning_to_word",
    ): CLASSICAL_CHINESE_MEANING_TO_WORD_MODEL_ID,
    ("french", "context"): FRENCH_CONTEXT_MODEL_ID,
    ("french", "word_to_meaning"): FRENCH_WORD_TO_MEANING_MODEL_ID,
    ("french", "meaning_to_word"): FRENCH_MEANING_TO_WORD_MODEL_ID,
    ("japanese", "context"): JAPANESE_CONTEXT_MODEL_ID,
    ("japanese", "word_to_meaning"): JAPANESE_WORD_TO_MEANING_MODEL_ID,
    ("japanese", "meaning_to_word"): JAPANESE_MEANING_TO_WORD_MODEL_ID,
    ("latin", "context"): LATIN_CONTEXT_MODEL_ID,
    ("latin", "word_to_meaning"): LATIN_WORD_TO_MEANING_MODEL_ID,
    ("latin", "meaning_to_word"): LATIN_MEANING_TO_WORD_MODEL_ID,
}
ENHANCED_MODEL_IDS = {
    ("english", "context"): 1190000001,
    ("english", "word_to_meaning"): 1190000002,
    ("english", "meaning_to_word"): 1190000003,
    ("classical_chinese", "context"): 1190000011,
    ("classical_chinese", "word_to_meaning"): 1190000012,
    ("classical_chinese", "meaning_to_word"): 1190000013,
    ("french", "context"): 1190000021,
    ("french", "word_to_meaning"): 1190000022,
    ("french", "meaning_to_word"): 1190000023,
    ("japanese", "context"): 1190000031,
    ("japanese", "word_to_meaning"): 1190000032,
    ("japanese", "meaning_to_word"): 1190000033,
}


@dataclass(frozen=True)
class CardType:
    key: str
    name: str
    model: genanki.Model
    language_key: str
    direction_key: str
    term_field: str
    enhanced: bool = False


def _create_card_type(language_key, direction_key, *, enhanced=False):
    language_name = LANGUAGE_NAMES[language_key]
    direction_name = DIRECTION_NAMES[direction_key]
    suffix = "_enhanced" if enhanced else ""
    return CardType(
        key=f"{language_key}_{direction_key}{suffix}",
        name=(
            f"Enhanced {language_name} {direction_name}"
            if enhanced
            else f"{language_name} {direction_name}"),
        model=(
            _create_enhanced_model(
                language_key,
                direction_key,
                ENHANCED_MODEL_IDS[(language_key, direction_key)])
            if enhanced
            else _create_model(
                language_key,
                direction_key,
                MODEL_IDS[(language_key, direction_key)])),
        language_key=language_key,
        direction_key=direction_key,
        term_field=TERM_FIELD_NAMES[language_key],
        enhanced=enhanced)


CARD_TYPES = {
    card_type.key: card_type
    for card_type in tuple(
        _create_card_type(language_key, direction_key)
        for language_key in LANGUAGE_NAMES
        for direction_key in DIRECTION_NAMES
    ) + tuple(
        _create_card_type(
            language_key,
            direction_key,
            enhanced=True)
        for language_key in (
            "english",
            "classical_chinese",
            "french",
            "japanese",
        )
        for direction_key in DIRECTION_NAMES)
}

SOURCE_SENTENCE_MODEL = _create_source_sentence_model()
ENHANCED_SOURCE_SENTENCE_MODEL = (
    _create_enhanced_source_sentence_model())

LEGACY_CARD_TYPE_KEYS = {
    "english_vocabulary": "english_context",
    "english_word_to_meaning": "english_word_to_meaning",
    "english_meaning_to_word": "english_meaning_to_word",
    "classical_chinese_vocabulary": "classical_chinese_context",
    "classical_chinese_native_vocabulary": "classical_chinese_context",
    "classical_chinese_dictionary_vocabulary": "classical_chinese_context",
    "classical_chinese_word_to_meaning": (
        "classical_chinese_word_to_meaning"),
    "classical_chinese_word_to_native_meaning": (
        "classical_chinese_word_to_meaning"),
    "classical_chinese_word_to_dictionary_meaning": (
        "classical_chinese_word_to_meaning"),
    "classical_chinese_meaning_to_word": (
        "classical_chinese_meaning_to_word"),
    "classical_chinese_native_meaning_to_word": (
        "classical_chinese_meaning_to_word"),
    "classical_chinese_dictionary_meaning_to_word": (
        "classical_chinese_meaning_to_word"),
    "french_vocabulary": "french_context",
    "french_native_vocabulary": "french_context",
    "french_dictionary_vocabulary": "french_context",
    "french_word_to_meaning": "french_word_to_meaning",
    "french_word_to_native_meaning": "french_word_to_meaning",
    "french_word_to_dictionary_meaning": "french_word_to_meaning",
    "french_meaning_to_word": "french_meaning_to_word",
    "french_native_meaning_to_word": "french_meaning_to_word",
    "french_dictionary_meaning_to_word": "french_meaning_to_word",
    "japanese_vocabulary": "japanese_context",
    "japanese_native_vocabulary": "japanese_context",
    "japanese_dictionary_vocabulary": "japanese_context",
    "japanese_word_to_meaning": "japanese_word_to_meaning",
    "japanese_word_to_native_meaning": "japanese_word_to_meaning",
    "japanese_word_to_dictionary_meaning": "japanese_word_to_meaning",
    "japanese_meaning_to_word": "japanese_meaning_to_word",
    "japanese_native_meaning_to_word": "japanese_meaning_to_word",
    "japanese_dictionary_meaning_to_word": "japanese_meaning_to_word",
    "latin_vocabulary": "latin_context",
    "latin_native_vocabulary": "latin_context",
    "latin_dictionary_vocabulary": "latin_context",
    "latin_word_to_meaning": "latin_word_to_meaning",
    "latin_word_to_native_meaning": "latin_word_to_meaning",
    "latin_word_to_dictionary_meaning": "latin_word_to_meaning",
    "latin_meaning_to_word": "latin_meaning_to_word",
    "latin_native_meaning_to_word": "latin_meaning_to_word",
    "latin_dictionary_meaning_to_word": "latin_meaning_to_word",
}

DEFAULT_CARD_TYPE_KEY = "english_context"
MODEL_ID = ENGLISH_CONTEXT_MODEL_ID


def get_card_type(card_type_key):
    canonical_key = LEGACY_CARD_TYPE_KEYS.get(
        card_type_key,
        card_type_key)
    try:
        return CARD_TYPES[canonical_key]
    except KeyError as error:
        raise ValueError(
            f"Unknown card type: {card_type_key}") from error


def get_direction_card_type(
        language_key,
        direction_key,
        *,
        enhanced=False):
    suffix = "_enhanced" if enhanced else ""
    return get_card_type(
        f"{language_key}_{direction_key}{suffix}")


def list_card_types(*, include_enhanced=False):
    """List registered direction models without breaking the legacy API."""
    return tuple(
        card_type
        for card_type in CARD_TYPES.values()
        if include_enhanced or not card_type.enhanced)


def create_deck(deck_id=DECK_ID, deck_name=DECK_NAME):
    return genanki.Deck(deck_id, deck_name)


my_deck = create_deck()

# Compatibility aliases for integrations which imported the original names.
ENGLISH_VOCABULARY_CARD_TYPE = get_direction_card_type("english", "context")
ENGLISH_WORD_TO_MEANING_CARD_TYPE = get_direction_card_type(
    "english", "word_to_meaning")
ENGLISH_MEANING_TO_WORD_CARD_TYPE = get_direction_card_type(
    "english", "meaning_to_word")
CLASSICAL_CHINESE_CARD_TYPE = get_direction_card_type(
    "classical_chinese", "context")
FRENCH_VOCABULARY_CARD_TYPE = get_direction_card_type("french", "context")
JAPANESE_VOCABULARY_CARD_TYPE = get_direction_card_type(
    "japanese", "context")
LATIN_VOCABULARY_CARD_TYPE = get_direction_card_type("latin", "context")
