import json
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import genanki


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
import prompt_builder
import templates


def language_pipeline(
        language_key,
        *,
        enabled=("context",),
        shared_fields=None,
        per_card_fields=None,
        share=True,
        separate=False):
    pipeline = pipeline_store.default_pipeline()
    settings = pipeline_store.get_language_settings(
        pipeline,
        language_key)
    shared_fields = (
        tuple(shared_fields)
        if shared_fields is not None
        else settings.shared_fields)
    per_card_fields = per_card_fields or {}
    cards = tuple(
        replace(
            card,
            enabled=card.direction_key in enabled,
            fields=tuple(per_card_fields.get(
                card.direction_key,
                card.fields)),
            target_deck=f"Deck::{card.direction_key}")
        for card in settings.cards)
    settings = replace(
        settings,
        cards=cards,
        target_deck=f"Deck::{language_key}",
        separate_target_decks=separate,
        share_field_settings=share,
        shared_fields=shared_fields)
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key=language_key)


class PipelineStoreTests(unittest.TestCase):
    def test_historical_variants_share_their_model_settings_category(self):
        self.assertEqual(
            tuple(
                language.name
                for language
                in pipeline_store.list_settings_languages()),
            (
                "English",
                "Classical Chinese",
                "French",
                "Japanese",
                "Latin",
            ))
        for language_key in (
                "classical_chinese",
                "classical_chinese_han",
                "classical_chinese_ming",
                "classical_chinese_warring_states",
                "classical_chinese_wang_bi"):
            self.assertEqual(
                pipeline_store.get_language(
                    language_key).model_language_key,
                "classical_chinese")
        for language_key in (
                "english",
                "middle_english",
                "old_english"):
            self.assertEqual(
                pipeline_store.get_language(
                    language_key).model_language_key,
                "english")

    def test_missing_settings_return_stable_default_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            pipelines = pipeline_store.load_pipelines(
                Path(directory) / "missing.json")

        self.assertEqual(
            pipelines,
            (pipeline_store.default_pipeline(),))
        self.assertEqual(
            pipelines[0].generated_deck_id,
            templates.DECK_ID)

    def test_current_settings_round_trip(self):
        french = language_pipeline(
            "french",
            enabled=("context", "meaning_to_word"),
            share=False,
            separate=True,
            per_card_fields={
                "context": (
                    pipeline_store.FieldSetting(
                        "translation",
                        "english"),
                    pipeline_store.FieldSetting(
                        "nuance",
                        "french"),
                ),
                "meaning_to_word": (
                    pipeline_store.FieldSetting(
                        "dictionary_meaning",
                        "japanese"),
                ),
            })
        second = pipeline_store.create_pipeline((french,))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipelines.json"
            pipeline_store.save_pipelines((french, second), path)
            loaded = pipeline_store.load_pipelines(path)
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, (french, second))
        self.assertEqual(
            raw["version"],
            pipeline_store.PIPELINE_CONFIG_VERSION)
        self.assertIn("shared_fields", raw["pipelines"][0])
        self.assertIn("cards", raw["pipelines"][0])
        self.assertEqual(
            pipeline_store.pipeline_from_mapping(
                pipeline_store.pipeline_to_mapping(french)),
            french)

    def test_unselected_field_language_is_saved_without_selecting_field(self):
        pipeline = pipeline_store.default_pipeline()
        settings = pipeline_store.get_language_settings(
            pipeline,
            "english")
        remembered = tuple(
            replace(
                field,
                target_language_key="latin")
            if field.field_key == "register"
            else field
            for field in settings.shared_field_languages)
        settings = replace(
            settings,
            shared_field_languages=remembered)
        pipeline = pipeline_store.replace_active_language_settings(
            pipeline,
            settings,
            (settings,))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipelines.json"
            pipeline_store.save_pipelines((pipeline,), path)
            loaded = pipeline_store.load_pipelines(path)[0]

        loaded_settings = pipeline_store.get_language_settings(
            loaded,
            "english")
        self.assertIn(
            pipeline_store.FieldSetting("register", "latin"),
            loaded_settings.shared_field_languages)
        self.assertNotIn(
            "register",
            {
                field.field_key
                for field in loaded_settings.shared_fields
            })
        self.assertNotIn(
            pipeline_store.FieldSetting("register", "latin"),
            pipeline_store.get_requested_field_settings(loaded))

    def test_version_six_settings_gain_remembered_language_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipelines.json"
            pipeline_store.save_pipelines(
                (pipeline_store.default_pipeline(),),
                path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["version"] = 6
            for item in raw["pipelines"]:
                item.pop("shared_field_languages", None)
                for card in item["cards"]:
                    card.pop("field_languages", None)
                for settings in item.get("language_settings", ()):
                    settings.pop("shared_field_languages", None)
                    for card in settings["cards"]:
                        card.pop("field_languages", None)
            path.write_text(json.dumps(raw), encoding="utf-8")

            loaded = pipeline_store.load_pipelines(path)[0]

        settings = pipeline_store.get_language_settings(
            loaded,
            loaded.language_key)
        self.assertEqual(
            len(settings.shared_field_languages),
            len(pipeline_store.list_field_options()))
        self.assertTrue(all(
            len(card.field_languages)
            == len(pipeline_store.list_field_options())
            for card in settings.cards))

    def test_legacy_card_types_migrate_to_directions_and_fields(self):
        legacy = {
            "version": 5,
            "pipelines": [{
                "pipeline_id": "legacy",
                "language_key": "french",
                "card_type_keys": [
                    "french_word_to_meaning",
                    "french_word_to_native_meaning",
                    "french_meaning_to_word",
                ],
                "target_deck": "French",
                "generated_deck_id": 1 << 30,
                "generated_deck_name": "Temporary French",
                "separate_target_decks": False,
                "language_settings": [],
            }],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipelines.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            pipeline = pipeline_store.load_pipelines(path)[0]

        self.assertEqual(pipeline.language_key, "french")
        self.assertEqual(
            tuple(
                card.direction_key
                for card in pipeline_store.get_enabled_cards(pipeline)),
            ("word_to_meaning", "meaning_to_word"))
        word_fields = pipeline.cards[1].fields
        self.assertIn(
            pipeline_store.FieldSetting("translation", "english"),
            word_fields)
        self.assertIn(
            pipeline_store.FieldSetting(
                "dictionary_meaning",
                "french"),
            word_fields)

    def test_version_one_singular_card_type_key_is_migrated(self):
        legacy = {
            "version": 1,
            "pipelines": [{
                "pipeline_id": "legacy",
                "language_key": "english",
                "card_type_key": "english_vocabulary",
                "target_deck": "English",
                "generated_deck_id": 1 << 30,
                "generated_deck_name": "Temporary English",
            }],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipelines.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            pipeline = pipeline_store.load_pipelines(path)[0]

        self.assertEqual(
            pipeline_store.get_card_type_keys(pipeline),
            ("english_context",))

    def test_translation_cannot_target_source_language(self):
        pipeline = language_pipeline(
            "french",
            shared_fields=(
                pipeline_store.FieldSetting(
                    "translation",
                    "french"),
            ))

        with self.assertRaisesRegex(
                ValueError,
                "Translation cannot target French"):
            pipeline_store.validate_pipelines((pipeline,))

    def test_historical_english_may_translate_into_modern_english(self):
        translation = (
            pipeline_store.FieldSetting(
                "translation",
                "english"),
        )
        for language_key in ("middle_english", "old_english"):
            with self.subTest(language_key=language_key):
                pipeline = language_pipeline(
                    language_key,
                    shared_fields=translation)
                self.assertEqual(
                    pipeline_store.validate_pipelines((pipeline,)),
                    (pipeline,))
                self.assertTrue(
                    pipeline_store.translation_target_allowed(
                        language_key,
                        "english"))

        modern = language_pipeline(
            "english",
            shared_fields=translation)
        with self.assertRaisesRegex(
                ValueError,
                "Translation cannot target English"):
            pipeline_store.validate_pipelines((modern,))

    def test_non_translation_field_may_target_source_language(self):
        pipeline = language_pipeline(
            "french",
            shared_fields=(
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "french"),
                pipeline_store.FieldSetting(
                    "register",
                    "french"),
            ))

        self.assertEqual(
            pipeline_store.validate_pipelines((pipeline,)),
            (pipeline,))

    def test_enabled_card_requires_at_least_one_effective_field(self):
        pipeline = language_pipeline(
            "english",
            shared_fields=())

        with self.assertRaisesRegex(
                ValueError,
                "at least one definition field"):
            pipeline_store.validate_pipelines((pipeline,))

    def test_three_directions_are_required_exactly_once(self):
        pipeline = replace(
            pipeline_store.default_pipeline(),
            cards=pipeline_store.default_pipeline().cards[:2])

        with self.assertRaisesRegex(
                ValueError,
                "each card direction exactly once"):
            pipeline_store.validate_pipelines((pipeline,))

    def test_shared_settings_apply_to_every_enabled_card(self):
        fields = (
            pipeline_store.FieldSetting(
                "dictionary_meaning",
                "latin"),
        )
        pipeline = language_pipeline(
            "english",
            enabled=("context", "word_to_meaning"),
            shared_fields=fields,
            share=True)
        settings = pipeline_store.get_language_settings(
            pipeline,
            "english")

        for card in pipeline_store.get_enabled_cards(pipeline):
            self.assertEqual(
                pipeline_store.get_effective_fields(settings, card),
                fields)
        self.assertEqual(
            pipeline_store.get_requested_field_settings(pipeline),
            fields)

    def test_per_card_settings_dedupe_only_identical_response_fields(self):
        english = pipeline_store.FieldSetting(
            "translation",
            "english")
        japanese = pipeline_store.FieldSetting(
            "translation",
            "japanese")
        nuance = pipeline_store.FieldSetting("nuance", "french")
        pipeline = language_pipeline(
            "french",
            enabled=(
                "context",
                "word_to_meaning",
                "meaning_to_word"),
            share=False,
            per_card_fields={
                "context": (english, nuance),
                "word_to_meaning": (english,),
                "meaning_to_word": (japanese,),
            })

        self.assertEqual(
            pipeline_store.get_requested_field_settings(pipeline),
            (english, nuance, japanese))
        self.assertEqual(
            tuple(
                pipeline_store.response_field_name(field)
                for field
                in pipeline_store.get_requested_field_settings(pipeline)),
            (
                "Translation (English)",
                "Nuance (French)",
                "Translation (Japanese)",
            ))

    def test_context_is_the_only_direction_that_requests_sentences(self):
        simple = language_pipeline(
            "english",
            enabled=("word_to_meaning",))
        detailed = language_pipeline(
            "english",
            enabled=("context", "word_to_meaning"))

        self.assertFalse(pipeline_store.requires_sentences(simple))
        self.assertTrue(pipeline_store.requires_sentences(detailed))

    def test_separate_decks_route_each_model(self):
        pipeline = language_pipeline(
            "latin",
            enabled=("word_to_meaning", "meaning_to_word"),
            separate=True)

        self.assertEqual(
            pipeline_store.get_model_target_decks(pipeline),
            (
                (
                    templates.get_direction_card_type(
                        "latin",
                        "word_to_meaning").model.name,
                    "Deck::word_to_meaning",
                ),
                (
                    templates.get_direction_card_type(
                        "latin",
                        "meaning_to_word").model.name,
                    "Deck::meaning_to_word",
                ),
            ))

    def test_separate_deck_requires_destination_for_enabled_card(self):
        pipeline = language_pipeline(
            "latin",
            enabled=("meaning_to_word",),
            separate=True)
        cards = tuple(
            replace(card, target_deck="")
            if card.direction_key == "meaning_to_word"
            else card
            for card in pipeline.cards)
        pipeline = replace(pipeline, cards=cards)

        with self.assertRaisesRegex(
                ValueError,
                "target Anki deck"):
            pipeline_store.validate_pipelines((pipeline,))

    def test_each_language_retains_its_preferences(self):
        pipeline = pipeline_store.default_pipeline()
        french = pipeline_store.get_language_settings(
            pipeline,
            "french")
        japanese = pipeline_store.get_language_settings(
            pipeline,
            "japanese")
        french = replace(
            french,
            target_deck="French",
            shared_fields=(
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "french"),
            ))
        japanese = replace(
            japanese,
            target_deck="Japanese")
        pipeline = pipeline_store.replace_active_language_settings(
            pipeline,
            french,
            (french, japanese))

        self.assertEqual(
            pipeline_store.get_language_settings(
                pipeline,
                "french").target_deck,
            "French")
        self.assertEqual(
            pipeline_store.get_language_settings(
                pipeline,
                "japanese").target_deck,
            "Japanese")

    def test_deck_cache_isolated_to_supplied_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anki_decks.json"
            saved = pipeline_store.save_anki_deck_cache(
                ("Vocabulary", "Vocabulary::French", "Vocabulary"),
                path)
            loaded = pipeline_store.load_anki_deck_cache(path)

        self.assertEqual(
            saved,
            ("Vocabulary", "Vocabulary::French"))
        self.assertEqual(loaded, saved)

    def test_new_pipeline_has_unique_persistent_deck_identity(self):
        first = pipeline_store.default_pipeline()
        second = pipeline_store.create_pipeline((first,))

        self.assertNotEqual(first.pipeline_id, second.pipeline_id)
        self.assertNotEqual(
            first.generated_deck_id,
            second.generated_deck_id)
        self.assertGreaterEqual(
            second.generated_deck_id,
            1 << 30)
        self.assertLess(second.generated_deck_id, 1 << 31)

    def test_duplicate_pipeline_and_deck_identities_are_rejected(self):
        first = pipeline_store.default_pipeline()
        duplicate = replace(
            pipeline_store.create_pipeline((first,)),
            pipeline_id=first.pipeline_id)
        with self.assertRaisesRegex(ValueError, "Pipeline IDs"):
            pipeline_store.validate_pipelines((first, duplicate))

        duplicate = replace(
            pipeline_store.create_pipeline((first,)),
            generated_deck_id=first.generated_deck_id)
        with self.assertRaisesRegex(ValueError, "deck IDs"):
            pipeline_store.validate_pipelines((first, duplicate))


class PromptComponentTests(unittest.TestCase):
    def _copy_components(self, root):
        target = Path(root) / "input" / "prompt_components"
        for source in (
                PROJECT_ROOT / "input" / "prompt_components").rglob("*"):
            if source.is_file():
                destination = target / source.relative_to(
                    PROJECT_ROOT / "input" / "prompt_components")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    source.read_text(encoding="utf-8"),
                    encoding="utf-8")

    def test_components_are_discovered_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            self._copy_components(directory)
            keys = {
                component.key
                for component
                in pipeline_store.discover_prompt_components(directory)
            }

        self.assertEqual(
            keys,
            {
                "core",
                "core_source_v9",
                "ending",
                "directions/context",
                "directions/context_arrays",
                "directions/context_arrays_v8",
                "directions/context_arrays_v9",
                "directions/context_classical_chinese",
                "directions/context_classical_chinese_arrays",
                "directions/sentence_translations",
                "directions/sentence_translation_arrays",
                "directions/sentence_translation_arrays_v9",
                "source/batch",
                "source/batch_v9",
                "source/context_examples",
                "source/context_examples_v5",
                "source/context_examples_v6",
                "source/context_examples_v7",
                "source/context_examples_v8",
                "source/context_examples_v9",
                "source/final_checks_v8",
                "source/final_checks_v9",
                "source/web_search",
                *{
                    f"fields/{field.key}"
                    for field in pipeline_store.list_field_options()
                },
                *{
                    f"languages/{language.key}"
                    for language in pipeline_store.list_languages()
                },
            })

    def test_prompt_contains_only_selected_linear_components(self):
        pipeline = language_pipeline(
            "french",
            enabled=("word_to_meaning", "meaning_to_word"),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "translation",
                    "english"),
                pipeline_store.FieldSetting(
                    "nuance",
                    "japanese"),
            ))
        text = prompt_builder.build_prompt(pipeline, PROJECT_ROOT)

        self.assertIn('For "Translation (English)"', text)
        self.assertIn('For "Nuance (Japanese)"', text)
        self.assertNotIn("four short", text)
        self.assertNotIn("Dictionary Meaning", text)
        self.assertNotIn("Pronunciation", text)

    def test_nuance_component_explicitly_allows_no_extra_nuance(self):
        pipeline = language_pipeline(
            "french",
            enabled=("word_to_meaning",),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "english"),
                pipeline_store.FieldSetting(
                    "nuance",
                    "english"),
            ))
        text = prompt_builder.build_prompt(pipeline, PROJECT_ROOT)
        normalized_text = " ".join(text.split())

        self.assertIn(
            "no meaningful nuance beyond the literal or dictionary meaning",
            normalized_text)
        self.assertIn(
            "return an empty string for this field",
            normalized_text)

    def test_dictionary_meaning_requires_an_explanation_not_a_translation(self):
        pipeline = language_pipeline(
            "french",
            enabled=("word_to_meaning",),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "translation",
                    "english"),
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "english"),
            ))
        text = prompt_builder.build_prompt(pipeline, PROJECT_ROOT)
        normalized_text = " ".join(text.split())

        self.assertIn(
            "explain in plain English what the selected sense actually means",
            normalized_text)
        self.assertIn(
            "understandable without knowing the source term",
            normalized_text)
        self.assertIn(
            "Do not give only translation equivalents, synonyms, or "
            "near-synonyms",
            normalized_text)
        self.assertIn(
            "do not repeat the translation field in different wording",
            normalized_text)

    def test_context_does_not_suppress_other_common_meanings(self):
        pipeline = language_pipeline(
            "english",
            enabled=("word_to_meaning",),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "english"),
            ))
        normalized_text = " ".join(
            prompt_builder.build_prompt(
                pipeline,
                PROJECT_ROOT).split())

        self.assertIn(
            "every genuinely disjoint, commonly used lexical sense that a "
            "learner is "
            "reasonably likely to encounter",
            normalized_text)
        self.assertIn(
            "even when the supplied context makes one particular sense "
            "unambiguous",
            normalized_text)
        self.assertIn(
            "Return the contextual sense first",
            normalized_text)
        self.assertIn(
            "Do not create separate entries for rare",
            normalized_text)
        self.assertIn(
            "not merely a different connotation, degree, register, "
            "implication, typical context, or other nuance",
            normalized_text)

    def test_context_adds_sentence_component_once(self):
        pipeline = language_pipeline(
            "latin",
            enabled=("context", "word_to_meaning"))
        text = prompt_builder.build_prompt(pipeline, PROJECT_ROOT)

        self.assertEqual(text.count("exactly four short"), 1)
        self.assertIn(
            "grammatically appropriate inflected form",
            " ".join(text.split()))
        self.assertIn(
            "wrap EVERY AND ONLY occurrence",
            " ".join(text.split()))
        self.assertIn("<strong>", text)
        self.assertIn(
            "leave that other occurrence completely untagged",
            " ".join(text.split()))
        self.assertIn(
            "VALID: She <strong>ran</strong> home",
            " ".join(text.split()))
        self.assertIn(
            "INVALID: <strong>She ran home.</strong>",
            " ".join(text.split()))
        self.assertIn(
            "ABSOLUTE OUTPUT REQUIREMENT—NOT OPTIONAL",
            text)
        self.assertIn(
            'four complete, natural English translations in the '
            '"Sentence Translations (English)" property',
            " ".join(text.split()))
        self.assertIn(
            "translation 1 must translate sentence 1",
            " ".join(text.split()))

    def test_classical_chinese_context_has_explicit_emphasis_objects(self):
        pipeline = language_pipeline(
            "classical_chinese_wang_bi",
            enabled=("context",))
        text = prompt_builder.build_prompt(pipeline, PROJECT_ROOT)

        self.assertIn(
            '"Sentences": "動善時。|知善時而動。|失善時則敗。|守其善時。"',
            text)
        self.assertIn(
            "用兵貴<strong>善時</strong>",
            text)
        normalized_text = " ".join(text.split())
        self.assertIn(
            "Generate a distinct set of sentences for every separate sense "
            "entry",
            normalized_text)
        self.assertIn(
            "infer unambiguously which particular sense is being used "
            "without seeing the definition",
            normalized_text)

    def test_classical_chinese_era_prompts_add_historical_guidance(self):
        ming = language_pipeline(
            "classical_chinese_ming",
            enabled=("word_to_meaning",),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "english"),
                pipeline_store.FieldSetting(
                    "register",
                    "english"),
            ))
        warring_states = language_pipeline(
            "classical_chinese_warring_states",
            enabled=("word_to_meaning",),
            shared_fields=(
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "english"),
                pipeline_store.FieldSetting(
                    "register",
                    "english"),
            ))

        ming_prompt = " ".join(
            prompt_builder.build_prompt(
                ming,
                PROJECT_ROOT).split())
        warring_prompt = " ".join(
            prompt_builder.build_prompt(
                warring_states,
                PROJECT_ROOT).split())

        self.assertIn(
            "Infer each sense from Ming usage",
            ming_prompt)
        self.assertIn(
            "particularly associated with the Ming period",
            ming_prompt)
        self.assertIn(
            "Infer each sense from pre-Qin and Warring States usage",
            warring_prompt)
        self.assertIn(
            "particular to the Warring States period",
            warring_prompt)
        for prompt in (ming_prompt, warring_prompt):
            self.assertIn(
                'do not use "archaic" when a narrower historical label '
                "is known",
                prompt)

    def test_daodejing_prompts_keep_recensions_and_registers_distinct(self):
        def rendered(language_key):
            pipeline = language_pipeline(
                language_key,
                enabled=("word_to_meaning",),
                shared_fields=(
                    pipeline_store.FieldSetting(
                        "dictionary_meaning",
                        "english"),
                    pipeline_store.FieldSetting(
                        "register",
                        "english"),
                ))
            return " ".join(
                prompt_builder.build_prompt(
                    pipeline,
                    PROJECT_ROOT).split())

        mawangdui = rendered("classical_chinese_han")
        wang_bi = rendered("classical_chinese_wang_bi")

        self.assertIn(
            "early Western Han form of the Laozi",
            mawangdui)
        self.assertIn(
            "Preserve manuscript and transcription forms such as 无, 亓",
            mawangdui)
        self.assertIn(
            "Mawangdui silk-manuscript or early Western Han recension",
            mawangdui)
        self.assertIn(
            "received Laozi text transmitted with Wang Bi",
            wang_bi)
        self.assertIn(
            "do not import wording from the Mawangdui silk manuscripts",
            wang_bi)

    def test_historical_english_prompts_preserve_stage_and_orthography(self):
        middle = language_pipeline(
            "middle_english",
            enabled=("word_to_meaning",),
            shared_fields=(
                pipeline_store.FieldSetting("translation", "english"),
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "english"),
                pipeline_store.FieldSetting("register", "english"),
            ))
        old = language_pipeline(
            "old_english",
            enabled=("word_to_meaning",),
            shared_fields=(
                pipeline_store.FieldSetting("translation", "english"),
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "english"),
                pipeline_store.FieldSetting("register", "english"),
            ))

        middle_prompt = " ".join(
            prompt_builder.build_prompt(
                middle,
                PROJECT_ROOT).split())
        old_prompt = " ".join(
            prompt_builder.build_prompt(
                old,
                PROJECT_ROOT).split())

        self.assertIn(
            "Middle English (approximately 1100–1500)",
            middle_prompt)
        self.assertIn(
            "late-fourteenth-century London/East Midlands usage",
            middle_prompt)
        self.assertIn(
            "preserving yogh, thorn, ash, diacritics",
            middle_prompt)
        self.assertIn(
            "Old English (approximately 450–1150)",
            old_prompt)
        self.assertIn(
            "preserving thorn, eth, ash, wynn, yogh",
            old_prompt)
        self.assertIn(
            "heroic verse such as Beowulf",
            old_prompt)
        for prompt in (middle_prompt, old_prompt):
            self.assertIn(
                "Copy the supplied source form exactly into the Word field",
                prompt)
            self.assertIn("reconstructed IPA form", prompt)
            self.assertIn(
                'For "Translation (English)"',
                prompt)

    def test_component_editor_saves_atomically_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            self._copy_components(directory)
            root = Path(directory)
            component = (
                root / "input" / "prompt_components" / "core")
            pipeline_store.save_prompt_component_text(
                component,
                "Replacement component",
                root)
            self.assertEqual(
                component.read_text(encoding="utf-8"),
                "Replacement component")
            self.assertEqual(
                list(component.parent.glob(".core.*")),
                [])

            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                pipeline_store.save_prompt_component_text(
                    component,
                    " ",
                    root)
            outside = root / "outside"
            outside.write_text("outside", encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError,
                    "must remain inside"):
                pipeline_store.save_prompt_component_text(
                    outside,
                    "replacement",
                    root)


class TemplateTests(unittest.TestCase):
    def test_registered_model_ids_are_unique_and_stable(self):
        card_types = templates.list_card_types()
        ids = [card_type.model.model_id for card_type in card_types]

        self.assertEqual(len(card_types), 15)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            templates.ENGLISH_CONTEXT_MODEL_ID,
            2092222676)
        self.assertEqual(
            templates.LATIN_MEANING_TO_WORD_MODEL_ID,
            2021716093)

    def test_every_language_has_exactly_three_directions(self):
        model_language_keys = {
            language.model_language_key
            for language in pipeline_store.list_languages()
        }
        for language_key in model_language_keys:
            card_types = [
                card_type
                for card_type in templates.list_card_types()
                if card_type.language_key == language_key
            ]
            self.assertEqual(
                {card_type.direction_key for card_type in card_types},
                {
                    "context",
                    "word_to_meaning",
                    "meaning_to_word",
                })

    def test_classical_chinese_eras_share_the_same_three_models(self):
        for language_key in (
                "classical_chinese_han",
                "classical_chinese_ming",
                "classical_chinese_warring_states",
                "classical_chinese_wang_bi"):
            pipeline = language_pipeline(
                language_key,
                enabled=(
                    "context",
                    "word_to_meaning",
                    "meaning_to_word"))
            self.assertEqual(
                pipeline_store.get_card_type_keys(pipeline),
                (
                    "classical_chinese_context",
                    "classical_chinese_word_to_meaning",
                    "classical_chinese_meaning_to_word",
                ))

    def test_historical_english_variants_share_the_english_models(self):
        for language_key in ("middle_english", "old_english"):
            pipeline = language_pipeline(
                language_key,
                enabled=(
                    "context",
                    "word_to_meaning",
                    "meaning_to_word"))
            self.assertEqual(
                pipeline_store.get_card_type_keys(pipeline),
                (
                    "english_context",
                    "english_word_to_meaning",
                    "english_meaning_to_word",
                ))

    def test_models_use_stable_superset_fields_and_readable_css(self):
        for card_type in templates.list_card_types():
            self.assertEqual(
                [field["name"] for field in card_type.model.fields],
                [
                    card_type.term_field,
                    "Sentences",
                    "Translation",
                    "Dictionary Meaning",
                    "Pronunciation",
                    "Part of Speech",
                    "Register",
                    "Nuance",
                    "Sentence Translations (English)",
                ])
            self.assertIn("font-size: 20px", card_type.model.css)
            self.assertIn("display: flex", card_type.model.css)
            self.assertIn("align-items: center", card_type.model.css)
            self.assertIn("justify-content: center", card_type.model.css)
            self.assertIn("min-height: 100vh", card_type.model.css)
            self.assertIn("font-weight: 700", card_type.model.css)

    def test_context_cards_accept_one_undelimited_source_passage(self):
        context_models = [
            card_type.model
            for card_type in templates.list_card_types()
            if card_type.direction_key == "context"
        ]

        self.assertTrue(context_models)
        for model in context_models:
            front = model.templates[0]["qfmt"]
            self.assertIn('raw.includes("|")', front)
            self.assertIn("[raw.trim()]", front)
            self.assertIn("candidateIndices", front)
            self.assertIn(
                "if (!candidateIndices.length) return",
                front)
            self.assertIn('const blockPrefix = "\\uE000S"', front)
            self.assertIn('.split(escapeMarker + "1").join("|")', front)
            self.assertIn("white-space: pre-wrap", model.css)
            self.assertIn("overflow-wrap: anywhere", model.css)

    def test_every_model_can_be_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, card_type in enumerate(
                    templates.list_card_types()):
                deck = genanki.Deck(
                    (1 << 30) + index,
                    f"Test {index}")
                deck.add_note(genanki.Note(
                    model=card_type.model,
                    fields=[
                        "term",
                        "one|two|three|four",
                        "translation",
                        "definition",
                        "pronunciation",
                        "part of speech",
                        "register",
                        "nuance",
                        "one|two|three|four",
                    ]))
                output = Path(directory) / f"{index}.apkg"
                genanki.Package(deck).write_to_file(output)
                with zipfile.ZipFile(output) as archive:
                    self.assertIn("collection.anki2", archive.namelist())

    def test_meaning_fields_are_vertical_and_conditionally_rendered(self):
        card_type = templates.get_direction_card_type(
            "french",
            "word_to_meaning")
        answer = card_type.model.templates[0]["afmt"]

        self.assertIn('class="definition-stack"', answer)
        self.assertIn("{{#Translation}}", answer)
        self.assertIn(
            '<strong class="definition-label">Translation</strong>',
            answer)
        self.assertIn("{{#Nuance}}", answer)


if __name__ == "__main__":
    unittest.main()
