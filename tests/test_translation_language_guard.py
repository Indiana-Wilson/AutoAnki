import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
import process_text


def configured_context_pipeline(language_key):
    pipeline = pipeline_store.default_pipeline()
    settings = pipeline_store.get_language_settings(
        pipeline,
        language_key)
    settings = replace(
        settings,
        cards=tuple(
            replace(
                card,
                enabled=card.direction_key == "context")
            for card in settings.cards),
        share_field_settings=True,
        shared_fields=(
            pipeline_store.FieldSetting(
                "dictionary_meaning",
                "english"),
        ))
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key=language_key)


def chinese_card():
    return {
        "Classical Chinese": "道",
        "Sentences": (
            "人能弘<strong>道</strong>。|"
            "上善若水，而水近<strong>道</strong>。|"
            "聞<strong>道</strong>者日損。|"
            "大道甚夷，而人好徑；此亦<strong>道</strong>也。"),
        "Sentence Translations (English)": (
            "People can extend the Way.|"
            "The highest good is like water and near the Way.|"
            "One who hears the Way diminishes daily.|"
            "The great Way is level, though people prefer paths."),
        "Dictionary Meaning (English)": (
            "The Way, path, or guiding order in the selected sense."),
    }


class SentenceTranslationLanguageGuardTests(unittest.TestCase):
    def test_reports_each_high_confidence_failure_at_its_item_path(self):
        pipeline = configured_context_pipeline(
            "classical_chinese_wang_bi")
        card = chinese_card()
        card["Sentence Translations (English)"] = (
            "人能弘道。|"
            "此乃善也。|"
            "The 道 here means a guiding principle.|"
            "Dao is retained as a loanword here.")

        report = process_text.inspect_generated_response(
            json.dumps({"cards": [card]}, ensure_ascii=False),
            pipeline)

        problems = [
            problem
            for problem in report["problems"]
            if problem["code"] == "sentence_translation_not_english"
        ]
        self.assertEqual(len(problems), 2)
        self.assertEqual(
            [problem["path"] for problem in problems],
            [
                '$.cards[0]["Sentence Translations (English)"][0]',
                '$.cards[0]["Sentence Translations (English)"][1]',
            ])
        self.assertEqual(
            [problem["actual"]["reason"] for problem in problems],
            [
                "exact_source_copy",
                "source_script_without_latin_text",
            ])
        self.assertTrue(all(
            not problem["overrideable"]
            for problem in problems))
        self.assertTrue(report["structurally_valid"])
        self.assertFalse(report["can_manually_accept"])

    def test_mixed_english_names_and_loanwords_are_allowed(self):
        pipeline = configured_context_pipeline(
            "classical_chinese_wang_bi")
        card = chinese_card()
        card["Sentence Translations (English)"] = (
            "The 道 here is the Way.|"
            "Dao is a retained loanword.|"
            "Confucius discusses 道 in this sentence.|"
            "Café 道 is a mixed Latin-script label.")

        report = process_text.inspect_generated_response(
            json.dumps({"cards": [card]}, ensure_ascii=False),
            pipeline)

        self.assertNotIn(
            "sentence_translation_not_english",
            {
                problem["code"]
                for problem in report["problems"]
            })

    def test_exact_source_copy_is_rejected_for_latin_script_source(self):
        pipeline = configured_context_pipeline("french")
        card = {
            "French": "table",
            "Sentences": (
                "La <strong>table</strong> est ronde.|"
                "Posez le livre sur la <strong>table</strong>.|"
                "Cette <strong>table</strong> est ancienne.|"
                "Nous mangeons à <strong>table</strong>."),
            "Sentence Translations (English)": (
                "La table est ronde.|"
                "Put the book on the table.|"
                "This table is old.|"
                "We eat at the table."),
            "Dictionary Meaning (English)": (
                "A piece of furniture with a flat top."),
        }

        report = process_text.inspect_generated_response(
            json.dumps({"cards": [card]}, ensure_ascii=False),
            pipeline)

        problem = next(
            problem
            for problem in report["problems"]
            if problem["code"] == "sentence_translation_not_english")
        self.assertEqual(
            problem["actual"]["reason"],
            "exact_source_copy")
        self.assertFalse(problem["overrideable"])

    def test_identical_text_remains_valid_for_english_source(self):
        card = {
            "Word": "astrolabe",
            "Sentences": (
                "One <strong>astrolabe</strong>.|"
                "Two <strong>astrolabes</strong>.|"
                "This <strong>astrolabe</strong>.|"
                "That <strong>astrolabe</strong>."),
            "Sentence Translations (English)": (
                "One astrolabe.|Two astrolabes.|"
                "This astrolabe.|That astrolabe."),
            "Dictionary Meaning (English)": (
                "An instrument formerly used to determine celestial "
                "positions."),
            "Pronunciation (English)": "/ˈæstrəleɪb/",
        }

        report = process_text.inspect_generated_response(
            json.dumps({"cards": [card]}))

        self.assertNotIn(
            "sentence_translation_not_english",
            {
                problem["code"]
                for problem in report["problems"]
            })


if __name__ == "__main__":
    unittest.main()
