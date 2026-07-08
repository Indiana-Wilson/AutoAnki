import genanki
import json
import templates
from openai import OpenAI
import os


def fetch_response(words):

    key = os.environ["OPENAI_API_KEY"]

    client = OpenAI(api_key=key)

    with open("input/prompt", "r") as file:
        prompt = file.read()

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt + words
    )

    result = response.output_text

    with open("output/response.json", "w") as file:
        file.write(result)

    with open("output/response_log", "a") as file:
        file.write("\n<break>\n" + result)

    return result


def fetch_response_file(file_path):
    fetch_response(open(file_path).read())


# Package the notes into a deck
def process_json_text(text):
    deck = templates.my_deck
    parser = json.loads(text)

    for word_def in parser:
        word = word_def['Word']
        sentences = word_def['Sentences']
        meaning = word_def['Meaning']
        pronunciation = word_def['Pronunciation']

        # Create a card
        my_note = genanki.Note(
            model=templates.my_model,
            fields=[word,sentences,meaning,pronunciation])

        deck.add_note(my_note)

    # Export the notes into a deck
    genanki.Package(deck).write_to_file('./output/output.apkg')
    print("Card deck successfully created.")


def process_json_file(file_path):
    process_json_text(open(file_path).read())




fetch_response_file("input/words")
process_json_file("output/response.json")


