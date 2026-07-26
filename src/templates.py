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
SOURCE_CONTEXT_BLOCK_PREFIX = "\ue000S"
SOURCE_CONTEXT_ESCAPE_MARKER = "\ue000"

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

.context-example {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  gap: 0.65em;
  max-width: 44em;
}

.context-term {
  max-width: 44em;
  font-weight: 700;
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


def _definition_stack():
    sections = []
    for _field_key, field_name in CONTENT_FIELDS:
        sections.append(
            f"""{{{{#{field_name}}}}}
<div class="definition-section">
  <strong class="definition-label">{field_name}</strong>
  <div>{{{{{field_name}}}}}</div>
</div>
{{{{/{field_name}}}}}""")
    return (
        '<div class="definition-stack">\n'
        + "\n".join(sections)
        + "\n</div>")


def _context_front(term_field):
    return f"""<div id="sentence-source" hidden>{{{{Sentences}}}}</div>
<div class="context-example">
  <div class="term context-term">{{{{{term_field}}}}}</div>
  <div id="sentence" class="sentence"></div>
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
  document.getElementById("sentence").innerHTML = sentences[chosenIndex];
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
        question = _definition_stack()
        answer = (
            "<div>{{FrontSide}}<hr>"
            f'<div class="term">{{{{{term_field}}}}}</div></div>')

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


@dataclass(frozen=True)
class CardType:
    key: str
    name: str
    model: genanki.Model
    language_key: str
    direction_key: str
    term_field: str


def _create_card_type(language_key, direction_key):
    language_name = LANGUAGE_NAMES[language_key]
    direction_name = DIRECTION_NAMES[direction_key]
    return CardType(
        key=f"{language_key}_{direction_key}",
        name=f"{language_name} {direction_name}",
        model=_create_model(
            language_key,
            direction_key,
            MODEL_IDS[(language_key, direction_key)]),
        language_key=language_key,
        direction_key=direction_key,
        term_field=TERM_FIELD_NAMES[language_key])


CARD_TYPES = {
    card_type.key: card_type
    for card_type in (
        _create_card_type(language_key, direction_key)
        for language_key in LANGUAGE_NAMES
        for direction_key in DIRECTION_NAMES
    )
}

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


def get_direction_card_type(language_key, direction_key):
    return get_card_type(f"{language_key}_{direction_key}")


def list_card_types():
    return tuple(CARD_TYPES.values())


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
