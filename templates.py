import genanki

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
  1607392319, # Has to be unique
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



# An example of how to create a card
# my_note = genanki.Note(
#   model=my_model,
#   fields=['word', 'sentences', 'meaning','pronunciation'])

my_deck = genanki.Deck(
  2059400110, # Has to be unique
  'Generated English Words')