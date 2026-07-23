import genanki
from dataclasses import dataclass

MODEL_ID = 1607392319
CLASSICAL_CHINESE_MODEL_ID = 1411027859
ENGLISH_WORD_TO_MEANING_MODEL_ID = 2133447792
ENGLISH_MEANING_TO_WORD_MODEL_ID = 1160831863
CLASSICAL_CHINESE_WORD_TO_MEANING_MODEL_ID = 1381223926
CLASSICAL_CHINESE_MEANING_TO_WORD_MODEL_ID = 2019574865
CLASSICAL_CHINESE_NATIVE_VOCABULARY_MODEL_ID = 1254697694
CLASSICAL_CHINESE_WORD_TO_NATIVE_MEANING_MODEL_ID = 1287778399
CLASSICAL_CHINESE_NATIVE_MEANING_TO_WORD_MODEL_ID = 1660745008
FRENCH_VOCABULARY_MODEL_ID = 1515378362
FRENCH_WORD_TO_MEANING_MODEL_ID = 1403940702
FRENCH_MEANING_TO_WORD_MODEL_ID = 1330075849
FRENCH_NATIVE_VOCABULARY_MODEL_ID = 1926905088
FRENCH_WORD_TO_NATIVE_MEANING_MODEL_ID = 1583626091
FRENCH_NATIVE_MEANING_TO_WORD_MODEL_ID = 1926381106
JAPANESE_VOCABULARY_MODEL_ID = 1583393271
JAPANESE_WORD_TO_MEANING_MODEL_ID = 1811106012
JAPANESE_MEANING_TO_WORD_MODEL_ID = 1626468105
JAPANESE_NATIVE_VOCABULARY_MODEL_ID = 1237338226
JAPANESE_WORD_TO_NATIVE_MEANING_MODEL_ID = 1445783107
JAPANESE_NATIVE_MEANING_TO_WORD_MODEL_ID = 1742985907
LATIN_VOCABULARY_MODEL_ID = 1356080652
LATIN_WORD_TO_MEANING_MODEL_ID = 1402054856
LATIN_MEANING_TO_WORD_MODEL_ID = 1114747532
LATIN_NATIVE_VOCABULARY_MODEL_ID = 1660541128
LATIN_WORD_TO_NATIVE_MEANING_MODEL_ID = 1878711721
LATIN_NATIVE_MEANING_TO_WORD_MODEL_ID = 2007408843
DECK_ID = 2059400110
DECK_NAME = 'Generated English Words'
CARD_CSS = '''.card {
  font-size: 22px;
}
'''

# The HTML/JavaScript that I use for the front side of the card
front_script = '''<div id="sentence" class="sentence" style="text-align: center;"></div>

<script>
(function () {
  const raw = `{{Sentences}}`;
  const sentences = raw
    .split("|")
    .map(s => s.trim())
    .filter(Boolean);

  const key = "anki-vocab-sentence-{{text:Word}}";

  let chosen = sessionStorage.getItem(key);

  if (!chosen || !sentences.includes(chosen)) {
    chosen = sentences[Math.floor(Math.random() * sentences.length)];
    sessionStorage.setItem(key, chosen);
  }

  document.getElementById("sentence").innerHTML = chosen;
})();

</script>
'''

# The HTML/JavaScript that I use for the backside of the card
back_script = '''<div style="text-align: center;">
{{FrontSide}}

<hr>

{{#Pronunciation}}
<div class="notes">
  <i>{{Pronunciation}}</i>
</div>
{{/Pronunciation}}

<div class="meaning">
  {{Meaning}}
</div>

<script>
(function () {
  const key = "anki-vocab-sentence-{{text:Word}}";
  sessionStorage.removeItem(key);
})();
</script>
</div>'''

# Essentially a genanki representation of a Card-type
my_model = genanki.Model(
  MODEL_ID, # Has to be unique
  'English Vocabulary',
  fields=[
    {'name': 'Word'},       # Each one of these lines defines a field of the card.
    {'name': 'Sentences'},
    {'name': 'Meaning'},
    {'name': 'Pronunciation'}
  ],
  templates=[
    {
      'name': 'Generated English Card',  # Name of the card-type in Anki
      'qfmt': front_script,              # The HTML code Anki uses for displaying the front of the card
      'afmt': back_script,               # The HTML code Anki uses for displaying the back of the card
    },
  ],
  css=CARD_CSS)

classical_chinese_model = genanki.Model(
  CLASSICAL_CHINESE_MODEL_ID,
  'Classical Chinese Vocabulary',
  fields=[
    {'name': 'Classical Chinese'},
    {'name': 'Sentences'},
    {'name': 'Meaning'},
    {'name': 'Pronunciation'}
  ],
  templates=[
    {
      'name': 'Generated Classical Chinese Card',
      'qfmt': front_script.replace(
        '{{text:Word}}',
        '{{text:Classical Chinese}}'),
      'afmt': back_script
        .replace('{{text:Word}}', '{{text:Classical Chinese}}'),
    },
  ],
  css=CARD_CSS)

english_word_to_meaning_model = genanki.Model(
  ENGLISH_WORD_TO_MEANING_MODEL_ID,
  'English Vocabulary - Word to Meaning',
  fields=[
    {'name': 'Word'},
    {'name': 'Meaning'},
  ],
  templates=[{
    'name': 'English Word to Meaning',
    'qfmt': '<div style="text-align: center;">{{Word}}</div>',
    'afmt': (
      '<div style="text-align: center;">'
      '{{FrontSide}}<hr>{{Meaning}}</div>'),
  }],
  css=CARD_CSS)

english_meaning_to_word_model = genanki.Model(
  ENGLISH_MEANING_TO_WORD_MODEL_ID,
  'English Vocabulary - Meaning to Word',
  fields=[
    {'name': 'Word'},
    {'name': 'Meaning'},
  ],
  templates=[{
    'name': 'English Meaning to Word',
    'qfmt': '<div style="text-align: center;">{{Meaning}}</div>',
    'afmt': (
      '<div style="text-align: center;">'
      '{{FrontSide}}<hr>{{Word}}</div>'),
  }],
  css=CARD_CSS)

classical_chinese_word_to_meaning_model = genanki.Model(
  CLASSICAL_CHINESE_WORD_TO_MEANING_MODEL_ID,
  'Classical Chinese Vocabulary - Word to Meaning',
  fields=[
    {'name': 'Classical Chinese'},
    {'name': 'Meaning'},
  ],
  templates=[{
    'name': 'Classical Chinese Word to Meaning',
    'qfmt': (
      '<div style="text-align: center;">'
      '{{Classical Chinese}}</div>'),
    'afmt': (
      '<div style="text-align: center;">'
      '{{FrontSide}}<hr>{{Meaning}}</div>'),
  }],
  css=CARD_CSS)

classical_chinese_meaning_to_word_model = genanki.Model(
  CLASSICAL_CHINESE_MEANING_TO_WORD_MODEL_ID,
  'Classical Chinese Vocabulary - Meaning to Word',
  fields=[
    {'name': 'Classical Chinese'},
    {'name': 'Meaning'},
  ],
  templates=[{
    'name': 'Classical Chinese Meaning to Word',
    'qfmt': '<div style="text-align: center;">{{Meaning}}</div>',
    'afmt': (
      '<div style="text-align: center;">'
      '{{FrontSide}}<hr>{{Classical Chinese}}</div>'),
  }],
  css=CARD_CSS)


def _create_detailed_model(
    model_id,
    model_name,
    card_name,
    term_field,
    definition_field,
):
  return genanki.Model(
    model_id,
    model_name,
    fields=[
      {'name': term_field},
      {'name': 'Sentences'},
      {'name': definition_field},
      {'name': 'Pronunciation'},
    ],
    templates=[{
      'name': card_name,
      'qfmt': front_script.replace(
        '{{text:Word}}',
        f'{{{{text:{term_field}}}}}'),
      'afmt': back_script
        .replace(
          '{{text:Word}}',
          f'{{{{text:{term_field}}}}}')
        .replace(
          '{{Meaning}}',
          f'{{{{{definition_field}}}}}'),
    }],
    css=CARD_CSS)


def _create_simple_model(
    model_id,
    model_name,
    card_name,
    term_field,
    definition_field,
    *,
    reverse=False,
):
  front_field = definition_field if reverse else term_field
  back_field = term_field if reverse else definition_field
  return genanki.Model(
    model_id,
    model_name,
    fields=[
      {'name': term_field},
      {'name': definition_field},
    ],
    templates=[{
      'name': card_name,
      'qfmt': (
        '<div style="text-align: center;">'
        f'{{{{{front_field}}}}}</div>'),
      'afmt': (
        '<div style="text-align: center;">'
        f'{{{{FrontSide}}}}<hr>{{{{{back_field}}}}}</div>'),
    }],
    css=CARD_CSS)


classical_chinese_native_vocabulary_model = _create_detailed_model(
  CLASSICAL_CHINESE_NATIVE_VOCABULARY_MODEL_ID,
  'Classical Chinese Vocabulary - Native Definition',
  'Classical Chinese Context to Native Definition',
  'Classical Chinese',
  'Native Meaning')

classical_chinese_word_to_native_meaning_model = _create_simple_model(
  CLASSICAL_CHINESE_WORD_TO_NATIVE_MEANING_MODEL_ID,
  'Classical Chinese Vocabulary - Word to Native Definition',
  'Classical Chinese Word to Native Definition',
  'Classical Chinese',
  'Native Meaning')

classical_chinese_native_meaning_to_word_model = _create_simple_model(
  CLASSICAL_CHINESE_NATIVE_MEANING_TO_WORD_MODEL_ID,
  'Classical Chinese Vocabulary - Native Definition to Word',
  'Classical Chinese Native Definition to Word',
  'Classical Chinese',
  'Native Meaning',
  reverse=True)

french_vocabulary_model = _create_detailed_model(
  FRENCH_VOCABULARY_MODEL_ID,
  'French Vocabulary - English Definition',
  'French Context to English Definition',
  'French',
  'Meaning')

french_word_to_meaning_model = _create_simple_model(
  FRENCH_WORD_TO_MEANING_MODEL_ID,
  'French Vocabulary - Word to English Definition',
  'French Word to English Definition',
  'French',
  'Meaning')

french_meaning_to_word_model = _create_simple_model(
  FRENCH_MEANING_TO_WORD_MODEL_ID,
  'French Vocabulary - English Definition to Word',
  'French English Definition to Word',
  'French',
  'Meaning',
  reverse=True)

french_native_vocabulary_model = _create_detailed_model(
  FRENCH_NATIVE_VOCABULARY_MODEL_ID,
  'French Vocabulary - Native Definition',
  'French Context to Native Definition',
  'French',
  'Native Meaning')

french_word_to_native_meaning_model = _create_simple_model(
  FRENCH_WORD_TO_NATIVE_MEANING_MODEL_ID,
  'French Vocabulary - Word to Native Definition',
  'French Word to Native Definition',
  'French',
  'Native Meaning')

french_native_meaning_to_word_model = _create_simple_model(
  FRENCH_NATIVE_MEANING_TO_WORD_MODEL_ID,
  'French Vocabulary - Native Definition to Word',
  'French Native Definition to Word',
  'French',
  'Native Meaning',
  reverse=True)

japanese_vocabulary_model = _create_detailed_model(
  JAPANESE_VOCABULARY_MODEL_ID,
  'Japanese Vocabulary - English Definition',
  'Japanese Context to English Definition',
  'Japanese',
  'Meaning')

japanese_word_to_meaning_model = _create_simple_model(
  JAPANESE_WORD_TO_MEANING_MODEL_ID,
  'Japanese Vocabulary - Word to English Definition',
  'Japanese Word to English Definition',
  'Japanese',
  'Meaning')

japanese_meaning_to_word_model = _create_simple_model(
  JAPANESE_MEANING_TO_WORD_MODEL_ID,
  'Japanese Vocabulary - English Definition to Word',
  'Japanese English Definition to Word',
  'Japanese',
  'Meaning',
  reverse=True)

japanese_native_vocabulary_model = _create_detailed_model(
  JAPANESE_NATIVE_VOCABULARY_MODEL_ID,
  'Japanese Vocabulary - Native Definition',
  'Japanese Context to Native Definition',
  'Japanese',
  'Native Meaning')

japanese_word_to_native_meaning_model = _create_simple_model(
  JAPANESE_WORD_TO_NATIVE_MEANING_MODEL_ID,
  'Japanese Vocabulary - Word to Native Definition',
  'Japanese Word to Native Definition',
  'Japanese',
  'Native Meaning')

japanese_native_meaning_to_word_model = _create_simple_model(
  JAPANESE_NATIVE_MEANING_TO_WORD_MODEL_ID,
  'Japanese Vocabulary - Native Definition to Word',
  'Japanese Native Definition to Word',
  'Japanese',
  'Native Meaning',
  reverse=True)

latin_vocabulary_model = _create_detailed_model(
  LATIN_VOCABULARY_MODEL_ID,
  'Latin Vocabulary - English Definition',
  'Latin Context to English Definition',
  'Latin',
  'Meaning')

latin_word_to_meaning_model = _create_simple_model(
  LATIN_WORD_TO_MEANING_MODEL_ID,
  'Latin Vocabulary - Word to English Definition',
  'Latin Word to English Definition',
  'Latin',
  'Meaning')

latin_meaning_to_word_model = _create_simple_model(
  LATIN_MEANING_TO_WORD_MODEL_ID,
  'Latin Vocabulary - English Definition to Word',
  'Latin English Definition to Word',
  'Latin',
  'Meaning',
  reverse=True)

latin_native_vocabulary_model = _create_detailed_model(
  LATIN_NATIVE_VOCABULARY_MODEL_ID,
  'Latin Vocabulary - Native Definition',
  'Latin Context to Native Definition',
  'Latin',
  'Native Meaning')

latin_word_to_native_meaning_model = _create_simple_model(
  LATIN_WORD_TO_NATIVE_MEANING_MODEL_ID,
  'Latin Vocabulary - Word to Native Definition',
  'Latin Word to Native Definition',
  'Latin',
  'Native Meaning')

latin_native_meaning_to_word_model = _create_simple_model(
  LATIN_NATIVE_MEANING_TO_WORD_MODEL_ID,
  'Latin Vocabulary - Native Definition to Word',
  'Latin Native Definition to Word',
  'Latin',
  'Native Meaning',
  reverse=True)


@dataclass(frozen=True)
class CardType:
  key: str
  name: str
  model: genanki.Model
  field_names: tuple[str, ...]
  identity_fields: tuple[str, ...]


ENGLISH_VOCABULARY_CARD_TYPE = CardType(
  key='english_vocabulary',
  name='English Vocabulary',
  model=my_model,
  field_names=('Word', 'Sentences', 'Meaning', 'Pronunciation'),
  identity_fields=('Word', 'Meaning'))

CLASSICAL_CHINESE_CARD_TYPE = CardType(
  key='classical_chinese_vocabulary',
  name='Classical Chinese Vocabulary',
  model=classical_chinese_model,
  field_names=(
    'Classical Chinese',
    'Sentences',
    'Meaning',
    'Pronunciation'),
  identity_fields=('Classical Chinese', 'Meaning'))

ENGLISH_WORD_TO_MEANING_CARD_TYPE = CardType(
  key='english_word_to_meaning',
  name='English Word to Meaning',
  model=english_word_to_meaning_model,
  field_names=('Word', 'Meaning'),
  identity_fields=('Word', 'Meaning'))

ENGLISH_MEANING_TO_WORD_CARD_TYPE = CardType(
  key='english_meaning_to_word',
  name='English Meaning to Word',
  model=english_meaning_to_word_model,
  field_names=('Word', 'Meaning'),
  identity_fields=('Word', 'Meaning'))

CLASSICAL_CHINESE_WORD_TO_MEANING_CARD_TYPE = CardType(
  key='classical_chinese_word_to_meaning',
  name='Classical Chinese Word to Meaning',
  model=classical_chinese_word_to_meaning_model,
  field_names=('Classical Chinese', 'Meaning'),
  identity_fields=('Classical Chinese', 'Meaning'))

CLASSICAL_CHINESE_MEANING_TO_WORD_CARD_TYPE = CardType(
  key='classical_chinese_meaning_to_word',
  name='Classical Chinese Meaning to Word',
  model=classical_chinese_meaning_to_word_model,
  field_names=('Classical Chinese', 'Meaning'),
  identity_fields=('Classical Chinese', 'Meaning'))

CLASSICAL_CHINESE_NATIVE_VOCABULARY_CARD_TYPE = CardType(
  key='classical_chinese_native_vocabulary',
  name='Classical Chinese Context to Native Definition',
  model=classical_chinese_native_vocabulary_model,
  field_names=(
    'Classical Chinese',
    'Sentences',
    'Native Meaning',
    'Pronunciation'),
  identity_fields=('Classical Chinese', 'Native Meaning'))

CLASSICAL_CHINESE_WORD_TO_NATIVE_MEANING_CARD_TYPE = CardType(
  key='classical_chinese_word_to_native_meaning',
  name='Classical Chinese Word to Native Definition',
  model=classical_chinese_word_to_native_meaning_model,
  field_names=('Classical Chinese', 'Native Meaning'),
  identity_fields=('Classical Chinese', 'Native Meaning'))

CLASSICAL_CHINESE_NATIVE_MEANING_TO_WORD_CARD_TYPE = CardType(
  key='classical_chinese_native_meaning_to_word',
  name='Classical Chinese Native Definition to Word',
  model=classical_chinese_native_meaning_to_word_model,
  field_names=('Classical Chinese', 'Native Meaning'),
  identity_fields=('Classical Chinese', 'Native Meaning'))

FRENCH_VOCABULARY_CARD_TYPE = CardType(
  key='french_vocabulary',
  name='French Context to English Definition',
  model=french_vocabulary_model,
  field_names=('French', 'Sentences', 'Meaning', 'Pronunciation'),
  identity_fields=('French', 'Meaning'))

FRENCH_WORD_TO_MEANING_CARD_TYPE = CardType(
  key='french_word_to_meaning',
  name='French Word to English Definition',
  model=french_word_to_meaning_model,
  field_names=('French', 'Meaning'),
  identity_fields=('French', 'Meaning'))

FRENCH_MEANING_TO_WORD_CARD_TYPE = CardType(
  key='french_meaning_to_word',
  name='French English Definition to Word',
  model=french_meaning_to_word_model,
  field_names=('French', 'Meaning'),
  identity_fields=('French', 'Meaning'))

FRENCH_NATIVE_VOCABULARY_CARD_TYPE = CardType(
  key='french_native_vocabulary',
  name='French Context to Native Definition',
  model=french_native_vocabulary_model,
  field_names=(
    'French',
    'Sentences',
    'Native Meaning',
    'Pronunciation'),
  identity_fields=('French', 'Native Meaning'))

FRENCH_WORD_TO_NATIVE_MEANING_CARD_TYPE = CardType(
  key='french_word_to_native_meaning',
  name='French Word to Native Definition',
  model=french_word_to_native_meaning_model,
  field_names=('French', 'Native Meaning'),
  identity_fields=('French', 'Native Meaning'))

FRENCH_NATIVE_MEANING_TO_WORD_CARD_TYPE = CardType(
  key='french_native_meaning_to_word',
  name='French Native Definition to Word',
  model=french_native_meaning_to_word_model,
  field_names=('French', 'Native Meaning'),
  identity_fields=('French', 'Native Meaning'))

JAPANESE_VOCABULARY_CARD_TYPE = CardType(
  key='japanese_vocabulary',
  name='Japanese Context to English Definition',
  model=japanese_vocabulary_model,
  field_names=('Japanese', 'Sentences', 'Meaning', 'Pronunciation'),
  identity_fields=('Japanese', 'Meaning'))

JAPANESE_WORD_TO_MEANING_CARD_TYPE = CardType(
  key='japanese_word_to_meaning',
  name='Japanese Word to English Definition',
  model=japanese_word_to_meaning_model,
  field_names=('Japanese', 'Meaning'),
  identity_fields=('Japanese', 'Meaning'))

JAPANESE_MEANING_TO_WORD_CARD_TYPE = CardType(
  key='japanese_meaning_to_word',
  name='Japanese English Definition to Word',
  model=japanese_meaning_to_word_model,
  field_names=('Japanese', 'Meaning'),
  identity_fields=('Japanese', 'Meaning'))

JAPANESE_NATIVE_VOCABULARY_CARD_TYPE = CardType(
  key='japanese_native_vocabulary',
  name='Japanese Context to Native Definition',
  model=japanese_native_vocabulary_model,
  field_names=(
    'Japanese',
    'Sentences',
    'Native Meaning',
    'Pronunciation'),
  identity_fields=('Japanese', 'Native Meaning'))

JAPANESE_WORD_TO_NATIVE_MEANING_CARD_TYPE = CardType(
  key='japanese_word_to_native_meaning',
  name='Japanese Word to Native Definition',
  model=japanese_word_to_native_meaning_model,
  field_names=('Japanese', 'Native Meaning'),
  identity_fields=('Japanese', 'Native Meaning'))

JAPANESE_NATIVE_MEANING_TO_WORD_CARD_TYPE = CardType(
  key='japanese_native_meaning_to_word',
  name='Japanese Native Definition to Word',
  model=japanese_native_meaning_to_word_model,
  field_names=('Japanese', 'Native Meaning'),
  identity_fields=('Japanese', 'Native Meaning'))

LATIN_VOCABULARY_CARD_TYPE = CardType(
  key='latin_vocabulary',
  name='Latin Context to English Definition',
  model=latin_vocabulary_model,
  field_names=('Latin', 'Sentences', 'Meaning', 'Pronunciation'),
  identity_fields=('Latin', 'Meaning'))

LATIN_WORD_TO_MEANING_CARD_TYPE = CardType(
  key='latin_word_to_meaning',
  name='Latin Word to English Definition',
  model=latin_word_to_meaning_model,
  field_names=('Latin', 'Meaning'),
  identity_fields=('Latin', 'Meaning'))

LATIN_MEANING_TO_WORD_CARD_TYPE = CardType(
  key='latin_meaning_to_word',
  name='Latin English Definition to Word',
  model=latin_meaning_to_word_model,
  field_names=('Latin', 'Meaning'),
  identity_fields=('Latin', 'Meaning'))

LATIN_NATIVE_VOCABULARY_CARD_TYPE = CardType(
  key='latin_native_vocabulary',
  name='Latin Context to Native Definition',
  model=latin_native_vocabulary_model,
  field_names=(
    'Latin',
    'Sentences',
    'Native Meaning',
    'Pronunciation'),
  identity_fields=('Latin', 'Native Meaning'))

LATIN_WORD_TO_NATIVE_MEANING_CARD_TYPE = CardType(
  key='latin_word_to_native_meaning',
  name='Latin Word to Native Definition',
  model=latin_word_to_native_meaning_model,
  field_names=('Latin', 'Native Meaning'),
  identity_fields=('Latin', 'Native Meaning'))

LATIN_NATIVE_MEANING_TO_WORD_CARD_TYPE = CardType(
  key='latin_native_meaning_to_word',
  name='Latin Native Definition to Word',
  model=latin_native_meaning_to_word_model,
  field_names=('Latin', 'Native Meaning'),
  identity_fields=('Latin', 'Native Meaning'))

CARD_TYPES = {
  ENGLISH_VOCABULARY_CARD_TYPE.key: ENGLISH_VOCABULARY_CARD_TYPE,
  ENGLISH_WORD_TO_MEANING_CARD_TYPE.key:
    ENGLISH_WORD_TO_MEANING_CARD_TYPE,
  ENGLISH_MEANING_TO_WORD_CARD_TYPE.key:
    ENGLISH_MEANING_TO_WORD_CARD_TYPE,
  CLASSICAL_CHINESE_CARD_TYPE.key: CLASSICAL_CHINESE_CARD_TYPE,
  CLASSICAL_CHINESE_WORD_TO_MEANING_CARD_TYPE.key:
    CLASSICAL_CHINESE_WORD_TO_MEANING_CARD_TYPE,
  CLASSICAL_CHINESE_MEANING_TO_WORD_CARD_TYPE.key:
    CLASSICAL_CHINESE_MEANING_TO_WORD_CARD_TYPE,
  CLASSICAL_CHINESE_NATIVE_VOCABULARY_CARD_TYPE.key:
    CLASSICAL_CHINESE_NATIVE_VOCABULARY_CARD_TYPE,
  CLASSICAL_CHINESE_WORD_TO_NATIVE_MEANING_CARD_TYPE.key:
    CLASSICAL_CHINESE_WORD_TO_NATIVE_MEANING_CARD_TYPE,
  CLASSICAL_CHINESE_NATIVE_MEANING_TO_WORD_CARD_TYPE.key:
    CLASSICAL_CHINESE_NATIVE_MEANING_TO_WORD_CARD_TYPE,
  FRENCH_VOCABULARY_CARD_TYPE.key:
    FRENCH_VOCABULARY_CARD_TYPE,
  FRENCH_WORD_TO_MEANING_CARD_TYPE.key:
    FRENCH_WORD_TO_MEANING_CARD_TYPE,
  FRENCH_MEANING_TO_WORD_CARD_TYPE.key:
    FRENCH_MEANING_TO_WORD_CARD_TYPE,
  FRENCH_NATIVE_VOCABULARY_CARD_TYPE.key:
    FRENCH_NATIVE_VOCABULARY_CARD_TYPE,
  FRENCH_WORD_TO_NATIVE_MEANING_CARD_TYPE.key:
    FRENCH_WORD_TO_NATIVE_MEANING_CARD_TYPE,
  FRENCH_NATIVE_MEANING_TO_WORD_CARD_TYPE.key:
    FRENCH_NATIVE_MEANING_TO_WORD_CARD_TYPE,
  JAPANESE_VOCABULARY_CARD_TYPE.key:
    JAPANESE_VOCABULARY_CARD_TYPE,
  JAPANESE_WORD_TO_MEANING_CARD_TYPE.key:
    JAPANESE_WORD_TO_MEANING_CARD_TYPE,
  JAPANESE_MEANING_TO_WORD_CARD_TYPE.key:
    JAPANESE_MEANING_TO_WORD_CARD_TYPE,
  JAPANESE_NATIVE_VOCABULARY_CARD_TYPE.key:
    JAPANESE_NATIVE_VOCABULARY_CARD_TYPE,
  JAPANESE_WORD_TO_NATIVE_MEANING_CARD_TYPE.key:
    JAPANESE_WORD_TO_NATIVE_MEANING_CARD_TYPE,
  JAPANESE_NATIVE_MEANING_TO_WORD_CARD_TYPE.key:
    JAPANESE_NATIVE_MEANING_TO_WORD_CARD_TYPE,
  LATIN_VOCABULARY_CARD_TYPE.key:
    LATIN_VOCABULARY_CARD_TYPE,
  LATIN_WORD_TO_MEANING_CARD_TYPE.key:
    LATIN_WORD_TO_MEANING_CARD_TYPE,
  LATIN_MEANING_TO_WORD_CARD_TYPE.key:
    LATIN_MEANING_TO_WORD_CARD_TYPE,
  LATIN_NATIVE_VOCABULARY_CARD_TYPE.key:
    LATIN_NATIVE_VOCABULARY_CARD_TYPE,
  LATIN_WORD_TO_NATIVE_MEANING_CARD_TYPE.key:
    LATIN_WORD_TO_NATIVE_MEANING_CARD_TYPE,
  LATIN_NATIVE_MEANING_TO_WORD_CARD_TYPE.key:
    LATIN_NATIVE_MEANING_TO_WORD_CARD_TYPE,
}

DEFAULT_CARD_TYPE_KEY = ENGLISH_VOCABULARY_CARD_TYPE.key


def get_card_type(card_type_key):
  try:
    return CARD_TYPES[card_type_key]
  except KeyError as error:
    raise ValueError(
      f'Unknown card type: {card_type_key}') from error


def list_card_types():
  return tuple(CARD_TYPES.values())



def create_deck(deck_id=DECK_ID, deck_name=DECK_NAME):
  """Create an empty deck for one generation run."""
  return genanki.Deck(deck_id, deck_name)


# Kept for compatibility with callers that use the original module-level deck.
my_deck = create_deck()
