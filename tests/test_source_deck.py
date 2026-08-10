import json
import hashlib
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
import source_deck
import templates
import enhanced_audio


def all_direction_french_pipeline(*, enhanced=False):
    pipeline = pipeline_store.default_pipeline()
    settings = pipeline_store.get_language_settings(
        pipeline,
        "french")
    fields = (
        pipeline_store.FieldSetting("translation", "english"),
        pipeline_store.FieldSetting(
            "dictionary_meaning",
            "english"),
    )
    settings = replace(
        settings,
        cards=tuple(
            replace(
                card,
                enabled=True,
                fields=fields,
                enhanced=enhanced)
            for card in settings.cards),
        share_field_settings=True,
        shared_fields=fields)
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key="french")


class FakeAudioService:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.requests = []
        self.artifacts = {}

    def synthesize_many(self, requests):
        result = []
        for request in requests:
            self.requests.append(request)
            identity = (
                f"{enhanced_audio.normalize_tts_text(request.text)}\0"
                f"{request.language}\0{request.role}")
            cache_key = hashlib.sha256(
                identity.encode("utf-8")).hexdigest()
            artifact = self.artifacts.get(cache_key)
            if artifact is None:
                path = self.root / f"{cache_key}.wav"
                path.write_bytes(b"RIFF" + cache_key.encode("ascii"))
                manifest_path = self.root / f"{cache_key}.json"
                manifest_path.write_text("{}", encoding="utf-8")
                artifact = enhanced_audio.AudioArtifact(
                    cache_key=cache_key,
                    path=path,
                    manifest_path=manifest_path,
                    media_filename=f"_test_tts_{cache_key[:16]}.wav",
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    byte_count=path.stat().st_size,
                    codec="wav",
                    encoder="pcm_s16le",
                    language=request.language,
                    role=request.role,
                    backend="fake",
                    model_id="fake-model",
                    model_revision="fake-revision",
                    voice_id="fake-neutral")
                self.artifacts[cache_key] = artifact
            result.append(artifact)
        return tuple(result)


def source_sentence_response():
    return {
        "cards": [
            {
                "French": "alpha",
                "Translation (English)": "A",
                "Dictionary Meaning (English)": "The first item.",
            },
            {
                "French": "alpha",
                "Translation (English)": "beginning",
                "Dictionary Meaning (English)": "An origin.",
            },
            {
                "French": "bêta",
                "Translation (English)": "B",
                "Dictionary Meaning (English)": "The second item.",
            },
        ],
        "source_contexts": [
            {
                "context_id": "context-one",
                "sentence_id": "sentence-one",
                "original_sentence": "Alpha seule.",
                "english_translation": "Alpha alone.",
                "nuance": "",
                "word_ranks": [1],
                "terms": ["alpha"],
                "ranked_terms": [{
                    "rank": 1,
                    "term": "alpha",
                }],
            },
            {
                "context_id": "context-two",
                "sentence_id": "sentence-two",
                "original_sentence": "Alpha bêta.",
                "english_translation": "Alpha and beta.",
                "nuance": "A cultural implication.",
                "word_ranks": [1, 2],
                "terms": ["alpha", "bêta"],
                "ranked_terms": [
                    {
                        "rank": 1,
                        "term": "alpha",
                    },
                    {
                        "rank": 2,
                        "term": "bêta",
                    },
                ],
            },
        ],
    }


def package_rows(package_path, directory):
    with zipfile.ZipFile(package_path) as package:
        package.extract("collection.anki2", path=directory)
    database = sqlite3.connect(
        Path(directory) / "collection.anki2")
    try:
        decks = json.loads(
            database.execute(
                "SELECT decks FROM col").fetchone()[0])
        return [
            {
                "due": due,
                "model_id": model_id,
                "fields": fields.split("\x1f"),
                "deck_name": decks[str(deck_id)]["name"],
            }
            for due, model_id, fields, deck_id
            in database.execute(
                "SELECT cards.due, notes.mid, notes.flds, cards.did "
                "FROM cards JOIN notes ON notes.id = cards.nid "
                "ORDER BY cards.due, cards.id")
        ]
    finally:
        database.close()


def package_note_guids(package_path, directory, model_id):
    with zipfile.ZipFile(package_path) as package:
        package.extract("collection.anki2", path=directory)
    database = sqlite3.connect(
        Path(directory) / "collection.anki2")
    try:
        return {
            guid
            for (guid,) in database.execute(
                "SELECT guid FROM notes WHERE mid = ?",
                (model_id,))
        }
    finally:
        database.close()


class SourceDeckTests(unittest.TestCase):
    def test_deck_identity_is_stable_and_source_specific(self):
        self.assertEqual(
            source_deck.source_deck_id("source-one"),
            source_deck.source_deck_id("source-one"))
        self.assertNotEqual(
            source_deck.source_deck_id("source-one"),
            source_deck.source_deck_id("source-two"))
        self.assertEqual(
            source_deck.source_deck_name("  Journey   to the West "),
            "Vocabulary from Journey to the West")

    def test_combined_response_is_packaged_into_source_named_deck(self):
        pipeline = pipeline_store.default_pipeline()
        response = {
            "cards": [
                {
                    "Word": "flourish",
                    "Sentences": (
                        "Plants <strong>flourish</strong> here.|"
                        "Arts <strong>flourish</strong> in peace.|"
                        "Trade can <strong>flourish</strong>."),
                    "Sentence Translations (English)": (
                        "Plants thrive here.|"
                        "Arts prosper in peace.|"
                        "Trade can thrive."),
                    "Dictionary Meaning (English)": (
                        "To grow or develop successfully."),
                    "Pronunciation (English)": "IPA: /ˈflʌrɪʃ/",
                },
                {
                    "Word": "fulfil",
                    "Sentences": (
                        "They <strong>fulfil</strong> the promise.|"
                        "They <strong>fulfil</strong> the need.|"
                        "We <strong>fulfil</strong> our duties."),
                    "Sentence Translations (English)": (
                        "They carry out the promise.|"
                        "They meet the need.|"
                        "We perform our duties."),
                    "Dictionary Meaning (English)": (
                        "To carry out or bring to completion."),
                    "Pronunciation (English)": "IPA: /fʊlˈfɪl/",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.apkg"
            package_path, count = source_deck.create_source_package(
                response,
                source_title="A Work",
                source_key="work-key",
                pipeline=pipeline,
                output_path=output)

            self.assertEqual(package_path, output)
            self.assertEqual(count, 2)
            self.assertTrue(output.is_file())
            with zipfile.ZipFile(output) as package:
                package.extract("collection.anki2", path=directory)
            database = sqlite3.connect(
                Path(directory) / "collection.anki2")
            try:
                due_positions = tuple(
                    row[0]
                    for row in database.execute(
                        "SELECT due FROM cards ORDER BY id"))
            finally:
                database.close()
            self.assertEqual(due_positions, (1, 2))

    def test_source_example_can_be_one_undelimited_text_block(self):
        pipeline = pipeline_store.default_pipeline()
        response = {
            "cards": [{
                "Word": "flourish",
                "Sentences": (
                    "The gardens <strong>flourish</strong> after rain. "
                    "Art and trade <strong>flourish</strong> here too."),
                "Sentence Translations (English)": (
                    "The gardens thrive after rain. "
                    "Art and trade prosper here too."),
                "Dictionary Meaning (English)": (
                    "To grow or develop successfully."),
                "Pronunciation (English)": "IPA: /ˈflʌrɪʃ/",
            }],
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source-context.apkg"
            package_path, count = source_deck.create_source_package(
                response,
                source_title="A Work",
                source_key="work-key",
                pipeline=pipeline,
                output_path=output,
                use_source_for_example_sentences=True)

        self.assertEqual(package_path, output)
        self.assertEqual(count, 1)

    def test_source_sentences_are_first_class_notes_in_combined_order(self):
        pipeline = all_direction_french_pipeline()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source-sentences.apkg"
            package_path, count = source_deck.create_source_package(
                source_sentence_response(),
                source_title="A Work",
                source_key="work-key",
                pipeline=pipeline,
                output_path=output,
                use_source_for_example_sentences=True)
            rows = package_rows(package_path, directory)

        self.assertEqual(count, 8)
        self.assertEqual(
            [row["due"] for row in rows],
            list(range(1, 9)))
        self.assertEqual(
            [row["model_id"] for row in rows],
            [
                templates.FRENCH_WORD_TO_MEANING_MODEL_ID,
                templates.FRENCH_MEANING_TO_WORD_MODEL_ID,
                templates.FRENCH_WORD_TO_MEANING_MODEL_ID,
                templates.FRENCH_MEANING_TO_WORD_MODEL_ID,
                templates.SOURCE_SENTENCE_MODEL_ID,
                templates.FRENCH_WORD_TO_MEANING_MODEL_ID,
                templates.FRENCH_MEANING_TO_WORD_MODEL_ID,
                templates.SOURCE_SENTENCE_MODEL_ID,
            ])
        self.assertTrue(all(
            row["fields"][1] == ""
            for row in rows
            if row["model_id"] != templates.SOURCE_SENTENCE_MODEL_ID))
        sentence_rows = [
            row
            for row in rows
            if row["model_id"] == templates.SOURCE_SENTENCE_MODEL_ID
        ]
        self.assertEqual(
            [row["fields"] for row in sentence_rows],
            [
                ["Alpha seule.", "Alpha alone.", ""],
                [
                    "Alpha bêta.",
                    "Alpha and beta.",
                    "A cultural implication.",
                ],
            ])

    def test_source_sentence_is_packaged_when_all_associated_words_are_known(
            self):
        pipeline = all_direction_french_pipeline()
        response = {
            "cards": [],
            "source_contexts": [{
                "context_id": "known-context",
                "sentence_id": "known-sentence",
                "original_sentence": "Alpha bêta.",
                "english_translation": "Alpha and beta.",
                "nuance": "",
                "word_ranks": [1, 2],
                "terms": ["alpha", "bêta"],
                "ranked_terms": [
                    {"rank": 1, "term": "alpha"},
                    {"rank": 2, "term": "bêta"},
                ],
            }],
            "expected_source_sentence_ids": ["known-sentence"],
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "all-known-source-sentence.apkg"
            package_path, count = source_deck.create_source_package(
                response,
                source_title="A Work",
                source_key="work-key",
                pipeline=pipeline,
                output_path=output,
                use_source_for_example_sentences=True)
            rows = package_rows(package_path, directory)

        self.assertEqual(count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["model_id"],
            templates.SOURCE_SENTENCE_MODEL_ID)
        self.assertEqual(
            rows[0]["fields"],
            ["Alpha bêta.", "Alpha and beta.", ""])

    def test_enhanced_source_package_contains_audio_models_and_media(self):
        pipeline = all_direction_french_pipeline(enhanced=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = FakeAudioService(root / "audio-cache")
            output = root / "enhanced-source.apkg"
            package_path, count = source_deck.create_source_package(
                source_sentence_response(),
                source_title="A Work",
                source_key="enhanced-work-key",
                pipeline=pipeline,
                output_path=output,
                use_source_for_example_sentences=True,
                audio_service=service,
                audio_media_directory=root / "package-media")
            rows = package_rows(package_path, root / "extracted")
            with zipfile.ZipFile(package_path) as archive:
                media = json.loads(
                    archive.read("media").decode("utf-8"))

        self.assertEqual(count, 8)
        self.assertEqual(
            [row["model_id"] for row in rows],
            [
                templates.ENHANCED_MODEL_IDS[
                    ("french", "word_to_meaning")],
                templates.ENHANCED_MODEL_IDS[
                    ("french", "meaning_to_word")],
                templates.ENHANCED_MODEL_IDS[
                    ("french", "word_to_meaning")],
                templates.ENHANCED_MODEL_IDS[
                    ("french", "meaning_to_word")],
                templates.ENHANCED_SOURCE_SENTENCE_MODEL_ID,
                templates.ENHANCED_MODEL_IDS[
                    ("french", "word_to_meaning")],
                templates.ENHANCED_MODEL_IDS[
                    ("french", "meaning_to_word")],
                templates.ENHANCED_SOURCE_SENTENCE_MODEL_ID,
            ])
        lexical_rows = [
            row
            for row in rows
            if row["model_id"] != templates.ENHANCED_SOURCE_SENTENCE_MODEL_ID
        ]
        self.assertTrue(all(
            row["fields"][9].startswith("[sound:_test_tts_")
            for row in lexical_rows))
        self.assertTrue(all(
            bool(row["fields"][13]) != bool(row["fields"][14])
            for row in lexical_rows))
        sentence_rows = [
            row
            for row in rows
            if row["model_id"] == templates.ENHANCED_SOURCE_SENTENCE_MODEL_ID
        ]
        self.assertTrue(all(
            row["fields"][3].startswith("[sound:_test_tts_")
            for row in sentence_rows))
        self.assertTrue(all(
            bool(row["fields"][4]) != bool(row["fields"][5])
            for row in sentence_rows))
        packaged_names = set(media.values())
        referenced_names = {
            field.removeprefix("[sound:").removesuffix("]")
            for row in rows
            for field in row["fields"]
            if field.startswith("[sound:")
        }
        self.assertEqual(packaged_names, referenced_names)

    def test_additional_source_sense_examples_follow_its_lexical_cards(self):
        pipeline = all_direction_french_pipeline()
        response = source_sentence_response()
        response["cards"][1].update({
            "Sentences": "Alpha part.|Alpha début.|Alpha origine.",
            "Sentence Translations (English)": (
                "Alpha leaves.|Alpha begins.|Alpha originates."),
        })

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source-additional-examples.apkg"
            package_path, count = source_deck.create_source_package(
                response,
                source_title="A Work",
                source_key="work-key",
                pipeline=pipeline,
                output_path=output,
                use_source_for_example_sentences=True)
            rows = package_rows(package_path, directory)

        self.assertEqual(count, 9)
        self.assertEqual(
            [row["model_id"] for row in rows[:6]],
            [
                templates.FRENCH_WORD_TO_MEANING_MODEL_ID,
                templates.FRENCH_MEANING_TO_WORD_MODEL_ID,
                templates.FRENCH_WORD_TO_MEANING_MODEL_ID,
                templates.FRENCH_MEANING_TO_WORD_MODEL_ID,
                templates.FRENCH_CONTEXT_MODEL_ID,
                templates.SOURCE_SENTENCE_MODEL_ID,
            ])
        self.assertTrue(all(
            row["fields"][1] == ""
            for row in rows[:2]))
        self.assertTrue(all(
            row["fields"][1]
            == (
                "<strong>Alpha</strong> part.|"
                "<strong>Alpha</strong> début.|"
                "<strong>Alpha</strong> origine.")
            for row in rows[2:5]))

    def test_source_sentence_packaging_rejects_a_missing_sidecar(self):
        pipeline = all_direction_french_pipeline()
        response = source_sentence_response()
        response["source_contexts"] = []

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ValueError,
                    "requires at least one validated source sentence"):
                source_deck.create_source_package(
                    response,
                    source_title="A Work",
                    source_key="work-key",
                    pipeline=pipeline,
                    output_path=Path(directory) / "missing.apkg",
                    use_source_for_example_sentences=True)

    def test_expected_sentence_identities_reject_a_partial_sidecar(self):
        pipeline = all_direction_french_pipeline()
        response = source_sentence_response()
        response["expected_source_sentence_ids"] = [
            "sentence-one",
            "sentence-two",
        ]
        response["source_contexts"] = response["source_contexts"][:1]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ValueError,
                    "source sentence data is incomplete"):
                source_deck.create_source_package(
                    response,
                    source_title="A Work",
                    source_key="work-key",
                    pipeline=pipeline,
                    output_path=Path(directory) / "partial.apkg",
                    use_source_for_example_sentences=True)

    def test_source_sentence_text_is_not_double_entity_escaped(self):
        pipeline = all_direction_french_pipeline()
        response = source_sentence_response()
        response["cards"] = response["cards"][:1]
        response["source_contexts"] = response["source_contexts"][:1]
        response["source_contexts"][0].update({
            "original_sentence": "Alpha &amp; bêta.",
            "english_translation": "Alpha &amp; beta.",
            "nuance": "A &amp; B.",
        })

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "entities.apkg"
            package_path, _count = source_deck.create_source_package(
                response,
                source_title="A Work",
                source_key="work-key",
                pipeline=pipeline,
                output_path=output,
                use_source_for_example_sentences=True)
            rows = package_rows(package_path, directory)

        sentence = next(
            row
            for row in rows
            if row["model_id"] == templates.SOURCE_SENTENCE_MODEL_ID)
        self.assertEqual(
            sentence["fields"],
            [
                "Alpha &amp; bêta.",
                "Alpha &amp; beta.",
                "A &amp; B.",
            ])

    def test_distinct_job_seeds_prevent_source_sentence_note_merging(self):
        pipeline = all_direction_french_pipeline()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_package, _count = source_deck.create_source_package(
                source_sentence_response(),
                source_title="A Work",
                source_key="work-key",
                pipeline=pipeline,
                output_path=root / "first.apkg",
                guid_seed="source-job:first",
                use_source_for_example_sentences=True)
            second_package, _count = source_deck.create_source_package(
                source_sentence_response(),
                source_title="A Work",
                source_key="work-key",
                pipeline=pipeline,
                output_path=root / "second.apkg",
                guid_seed="source-job:second",
                use_source_for_example_sentences=True)
            first_guids = package_note_guids(
                first_package,
                root / "first",
                templates.SOURCE_SENTENCE_MODEL_ID)
            second_guids = package_note_guids(
                second_package,
                root / "second",
                templates.SOURCE_SENTENCE_MODEL_ID)

        self.assertEqual(len(first_guids), 2)
        self.assertEqual(len(second_guids), 2)
        self.assertTrue(first_guids.isdisjoint(second_guids))

    def test_separate_source_decks_have_independent_due_sequences(self):
        pipeline = all_direction_french_pipeline()
        response = source_sentence_response()
        response["cards"] = response["cards"][:1]
        response["source_contexts"] = response["source_contexts"][:1]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "separate-source.apkg"
            package_path, count = source_deck.create_source_package(
                response,
                source_title="A Work",
                source_key="work-key",
                pipeline=pipeline,
                output_path=output,
                use_source_for_example_sentences=True,
                separate_source_decks=True)
            rows = package_rows(package_path, directory)

        self.assertEqual(count, 3)
        self.assertEqual(
            {row["due"] for row in rows},
            {1})
        self.assertEqual(
            {row["deck_name"] for row in rows},
            {
                "Vocabulary from A Work::Sentence to Meaning",
                "Vocabulary from A Work::Word to Meaning",
                "Vocabulary from A Work::Meaning to Word",
            })

    def test_generated_examples_order_directions_within_each_word(self):
        pipeline = all_direction_french_pipeline()
        settings = pipeline_store.get_language_settings(
            pipeline,
            "french")
        reversed_settings = replace(
            settings,
            cards=tuple(reversed(settings.cards)))
        pipeline = pipeline_store.replace_active_language_settings(
            pipeline,
            reversed_settings,
            (reversed_settings,),
            active_language_key="french")
        response = {
            "cards": [
                {
                    "French": "alpha",
                    "Sentences": "Alpha un.|Alpha deux.|Alpha trois.",
                    "Sentence Translations (English)": (
                        "Alpha one.|Alpha two.|Alpha three."),
                    "Translation (English)": "A",
                    "Dictionary Meaning (English)": "The first item.",
                },
                {
                    "French": "bêta",
                    "Sentences": "Bêta un.|Bêta deux.|Bêta trois.",
                    "Sentence Translations (English)": (
                        "Beta one.|Beta two.|Beta three."),
                    "Translation (English)": "B",
                    "Dictionary Meaning (English)": "The second item.",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated-order.apkg"
            package_path, count = source_deck.create_source_package(
                response,
                source_title="A Work",
                source_key="work-key",
                pipeline=pipeline,
                output_path=output)
            rows = package_rows(package_path, directory)

        self.assertEqual(count, 6)
        self.assertEqual(
            [row["model_id"] for row in rows],
            [
                templates.FRENCH_CONTEXT_MODEL_ID,
                templates.FRENCH_WORD_TO_MEANING_MODEL_ID,
                templates.FRENCH_MEANING_TO_WORD_MODEL_ID,
                templates.FRENCH_CONTEXT_MODEL_ID,
                templates.FRENCH_WORD_TO_MEANING_MODEL_ID,
                templates.FRENCH_MEANING_TO_WORD_MODEL_ID,
            ])

    def test_legacy_source_response_can_be_packaged_without_translations(self):
        pipeline = pipeline_store.default_pipeline()
        response = {
            "cards": [{
                "Word": "flourish",
                "Sentences": (
                    "Plants <strong>flourish</strong> here.|"
                    "Arts <strong>flourish</strong> in peace.|"
                    "Trade can <strong>flourish</strong>."),
                "Dictionary Meaning (English)": (
                    "To grow or develop successfully."),
                "Pronunciation (English)": "IPA: /ˈflʌrɪʃ/",
            }],
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "legacy-source.apkg"
            package_path, count = source_deck.create_source_package(
                response,
                source_title="A Work",
                source_key="legacy-work-key",
                pipeline=pipeline,
                output_path=output,
                require_sentence_translations=False)

        self.assertEqual(package_path, output)
        self.assertEqual(count, 1)

    def test_import_uses_standalone_workflow(self):
        importer = MagicMock(return_value=True)
        client = MagicMock()

        result = source_deck.import_source_package(
            "/tmp/source.apkg",
            client=client,
            importer=importer)

        self.assertTrue(result)
        importer.assert_called_once_with(
            "/tmp/source.apkg",
            client=client)


if __name__ == "__main__":
    unittest.main()
