import genanki
from dataclasses import dataclass

MODEL_ID = 1607392319
CLASSICAL_CHINESE_MODEL_ID = 1411027859
ENGLISH_WORD_TO_MEANING_MODEL_ID = 2133447792
ENGLISH_MEANING_TO_WORD_MODEL_ID = 1160831863
CLASSICAL_CHINESE_WORD_TO_MEANING_MODEL_ID = 1381223926
CLASSICAL_CHINESE_MEANING_TO_WORD_MODEL_ID = 2019574865
DECK_ID = 2059400110
DECK_NAME = 'Generated English Words'

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
  ])

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
  ])

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
  }])

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
  }])

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
  }])

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
  }])


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
