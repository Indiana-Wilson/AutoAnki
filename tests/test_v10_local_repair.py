"""Focused tests for conservative compact-response local repair."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
import process_text
from source_generation.local_repair import repair_compact_response
from source_generation.jobs import GenerationJobStore
from source_generation.validation import (
    ValidatedPipelineResponse,
    inspect_pipeline_response,
    make_pipeline_response_validator,
)
from tests.test_v8_schema_contract import detailed_context_pipeline


def context_pipeline(language_key):
    pipeline = pipeline_store.default_pipeline()
    settings = pipeline_store.get_language_settings(
        pipeline,
        language_key)
    selected_fields = (
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
                enabled=card.direction_key == "context",
                fields=selected_fields)
            for card in settings.cards),
        share_field_settings=True,
        shared_fields=selected_fields)
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key=language_key)


def chunk(*surfaces):
    context_texts = tuple(
        f"例{surface}文。"
        for surface in surfaces)
    words = tuple(
        SimpleNamespace(
            rank=index,
            surface=surface,
            context_id=f"context-{index}",
            start_offset=context_texts[index - 1].index(surface),
            end_offset=(
                context_texts[index - 1].index(surface)
                + len(surface)))
        for index, surface in enumerate(surfaces, start=1))
    contexts = tuple(
        SimpleNamespace(
            context_id=word.context_id,
            text=context_texts[index - 1],
            start_offset=0)
        for index, word in enumerate(words, start=1))
    return SimpleNamespace(
        words=words,
        contexts=contexts)


class CompactLocalRepairTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = detailed_context_pipeline()
        self.chunk = chunk("難易", "知")
        self.lexical = {
            "Translation (English)": "value",
            "Dictionary Meaning (English)": "A useful meaning.",
            "Pronunciation (English)": "reading",
            "Part of Speech (English)": "noun",
            "Register (English)": "literary",
            "Nuance (English)": "",
        }

    def sense(self, *, sentences, translation="value"):
        return {
            **self.lexical,
            "Translation (English)": translation,
            "Sentences": sentences,
            "Sentence Translations (English)": [
                "First.",
                "Second.",
                "Third.",
            ],
        }

    def payload(self):
        return {
            process_text.SOURCE_TERM_RESULTS_KEY: [{
                "rank": 2,
                "contextual_sense": {
                    **self.lexical,
                    "Translation (English)": "know",
                },
                "additional_senses": [self.sense(
                    translation="understand",
                    sentences=[
                        "<strong>知</strong>一。",
                        "<strong>知</strong>二。",
                        "<strong>知</strong>三。",
                    ])],
            }, {
                "rank": 1,
                "contextual_sense": {
                    **self.lexical,
                    "Translation (English)": "difficult and easy",
                },
                "additional_senses": [
                    self.sense(
                        translation="difficult",
                        sentences=[
                            "難易相較，而事甚<strong>難</strong>。",
                            "學道不<strong>難</strong>。",
                            "守靜最<strong>難</strong>。",
                            "此策非<strong>難</strong>。",
                        ]),
                ],
            }],
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: [{
                "context_id": "context-2",
                "translation": "<b>Second context.</b>",
            }, {
                "context_id": "context-1",
                "translation": "First context.",
            }],
        }

    def test_repairs_structure_html_and_component_only_optional_sense(self):
        payload = self.payload()
        payload["unrequested"] = "ignored"
        payload["term_results"].append({
            "rank": 999,
            "contextual_sense": dict(self.lexical),
            "additional_senses": [],
        })

        result = repair_compact_response(
            json.dumps(payload, ensure_ascii=False),
            self.pipeline,
            self.chunk)

        self.assertTrue(result.changed)
        repaired = json.loads(result.candidate_raw_text)
        self.assertEqual(
            [item["rank"] for item in repaired["term_results"]],
            [1, 2])
        self.assertEqual(
            [item["context_id"]
             for item in repaired["source_context_translations"]],
            ["context-1", "context-2"])
        self.assertNotIn("unrequested", repaired)
        self.assertEqual(
            repaired["term_results"][0]["additional_senses"],
            [])
        self.assertEqual(
            repaired["term_results"][1]["additional_senses"][0][
                "Sentences"],
            ["知一。", "知二。", "知三。"])
        self.assertEqual(
            repaired["source_context_translations"][1]["translation"],
            "Second context.")
        rules = {
            change["rule_id"]
            for change in result.changes
        }
        self.assertIn("discard_component_only_sense", rules)
        self.assertIn("remove_model_html", rules)
        self.assertIn("restore_frozen_identity_order", rules)
        self.assertEqual(
            result.audit_record()["candidate_sha256"],
            result.candidate_sha256)

    def test_discards_translation_for_retained_but_non_owned_context(self):
        request_chunk = chunk("難易", "知")
        request_chunk.source_context_translation_ids = ("context-1",)

        result = repair_compact_response(
            json.dumps(self.payload(), ensure_ascii=False),
            self.pipeline,
            request_chunk)
        repaired = json.loads(result.candidate_raw_text)

        self.assertEqual(
            repaired["source_context_translations"],
            [{
                "context_id": "context-1",
                "translation": "First context.",
            }])
        self.assertIn(
            "drop_unrequested_identity",
            {
                change["rule_id"]
                for change in result.changes
            })

    def test_discards_redundant_translation_memory_echo(self):
        result = repair_compact_response(
            json.dumps(self.payload(), ensure_ascii=False),
            self.pipeline,
            self.chunk,
            remembered_source_context_ids={"context-2"})
        repaired = json.loads(result.candidate_raw_text)

        self.assertEqual(
            repaired["source_context_translations"],
            [{
                "context_id": "context-1",
                "translation": "First context.",
            }])
        self.assertIn(
            "drop_unrequested_identity",
            {
                change["rule_id"]
                for change in result.changes
            })

    def test_repair_is_idempotent(self):
        first = repair_compact_response(
            json.dumps(self.payload(), ensure_ascii=False),
            self.pipeline,
            self.chunk)
        second = repair_compact_response(
            first.candidate_raw_text,
            self.pipeline,
            self.chunk)

        self.assertFalse(second.changed)
        self.assertEqual(second.changes, ())
        self.assertEqual(
            second.candidate_raw_text,
            first.candidate_raw_text)

    def test_repairs_safe_html_in_every_lexical_string_field(self):
        payload = self.payload()
        contextual = payload["term_results"][0]["contextual_sense"]
        additional = payload["term_results"][0][
            "additional_senses"][0]
        for sense in (contextual, additional):
            sense["Translation (English)"] = (
                "<b>" + sense["Translation (English)"] + "</b>")
            sense["Dictionary Meaning (English)"] = (
                "<em>" + sense["Dictionary Meaning (English)"] + "</em>")
            sense["Pronunciation (English)"] = (
                "<span>" + sense["Pronunciation (English)"] + "</span>")
            sense["Part of Speech (English)"] = (
                "<u>" + sense["Part of Speech (English)"] + "</u>")
            sense["Register (English)"] = (
                "<i>" + sense["Register (English)"] + "</i>")
            sense["Nuance (English)"] = (
                "<mark>" + sense["Nuance (English)"] + "</mark>")

        first = repair_compact_response(
            json.dumps(payload, ensure_ascii=False),
            self.pipeline,
            self.chunk)
        repaired = json.loads(first.candidate_raw_text)
        repaired_result = next(
            result
            for result in repaired["term_results"]
            if result["rank"] == 2)
        repaired_contextual = repaired_result["contextual_sense"]
        repaired_additional = repaired_result["additional_senses"][0]

        for sense in (repaired_contextual, repaired_additional):
            for field_name in self.lexical:
                self.assertNotIn("<", sense[field_name])
                self.assertNotIn(">", sense[field_name])
        second = repair_compact_response(
            first.candidate_raw_text,
            self.pipeline,
            self.chunk)
        self.assertFalse(second.changed)

    def test_does_not_flatten_unsafe_or_content_bearing_html(self):
        payload = self.payload()
        payload["term_results"][0]["contextual_sense"][
            "Translation (English)"] = (
                "<script>window.bad = true</script>know")
        payload["term_results"][0]["contextual_sense"][
            "Dictionary Meaning (English)"] = (
                "<ruby>知<rt>zhī</rt></ruby> means know.")

        result = repair_compact_response(
            json.dumps(payload, ensure_ascii=False),
            self.pipeline,
            self.chunk)
        repaired = json.loads(result.candidate_raw_text)
        contextual = next(
            item
            for item in repaired["term_results"]
            if item["rank"] == 2)["contextual_sense"]

        self.assertIn("<script>", contextual["Translation (English)"])
        self.assertIn("<ruby>", contextual[
            "Dictionary Meaning (English)"])
        report = inspect_pipeline_response(
            json.dumps(payload, ensure_ascii=False),
            self.pipeline,
            self.chunk,
            use_compact_source_results=True,
            use_local_example_emphasis=True)
        self.assertIn(
            "generated_field_contains_html",
            {problem["code"] for problem in report["problems"]})

    def test_inflecting_language_does_not_prune_absent_exact_form(self):
        pipeline = context_pipeline("french")
        request_chunk = chunk("courir")
        payload = {
            "term_results": [{
                "rank": 1,
                "contextual_sense": {
                    "Translation (English)": "run",
                    "Dictionary Meaning (English)": "To move quickly.",
                },
                "additional_senses": [{
                    "Translation (English)": "operate",
                    "Dictionary Meaning (English)": (
                        "To function or manage something."),
                    "Sentences": [
                        "Elle court vite.",
                        "Nous courons ensemble.",
                        "Ils couraient hier.",
                    ],
                    "Sentence Translations (English)": [
                        "She runs quickly.",
                        "We run together.",
                        "They were running yesterday.",
                    ],
                }],
            }],
            "source_context_translations": [{
                "context_id": "context-1",
                "translation": "She wants to run.",
            }],
        }

        result = repair_compact_response(
            json.dumps(payload),
            pipeline,
            request_chunk)

        repaired = json.loads(result.candidate_raw_text)
        self.assertEqual(
            len(repaired["term_results"][0]["additional_senses"]),
            1)

    def test_discards_only_wholly_unexemplified_exact_form_sense(self):
        payload = self.payload()
        result = next(
            item
            for item in payload["term_results"]
            if item["rank"] == 2)
        result["additional_senses"][0]["Sentences"] = [
            "曉一。",
            "曉二。",
            "曉三。",
        ]

        discarded = repair_compact_response(
            json.dumps(payload, ensure_ascii=False),
            self.pipeline,
            self.chunk)
        discarded_result = next(
            item
            for item in json.loads(
                discarded.candidate_raw_text)["term_results"]
            if item["rank"] == 2)

        self.assertEqual(discarded_result["additional_senses"], [])
        self.assertIn(
            "discard_absent_complete_term_sense",
            {change["rule_id"] for change in discarded.changes})

        payload = self.payload()
        result = next(
            item
            for item in payload["term_results"]
            if item["rank"] == 2)
        result["additional_senses"][0]["Sentences"] = [
            "知一。",
            "曉二。",
            "曉三。",
        ]
        retained = repair_compact_response(
            json.dumps(payload, ensure_ascii=False),
            self.pipeline,
            self.chunk)
        retained_result = next(
            item
            for item in json.loads(
                retained.candidate_raw_text)["term_results"]
            if item["rank"] == 2)

        self.assertEqual(len(retained_result["additional_senses"]), 1)
        self.assertNotIn(
            "discard_absent_complete_term_sense",
            {change["rule_id"] for change in retained.changes})

    def test_does_not_guess_conflicting_duplicate_identity(self):
        payload = self.payload()
        duplicate = json.loads(json.dumps(payload["term_results"][0]))
        duplicate["contextual_sense"]["Translation (English)"] = "conflict"
        payload["term_results"].append(duplicate)

        result = repair_compact_response(
            json.dumps(payload, ensure_ascii=False),
            self.pipeline,
            self.chunk)

        repaired = json.loads(result.candidate_raw_text)
        self.assertEqual(
            [item["rank"] for item in repaired["term_results"]].count(2),
            2)

    def test_does_not_discard_polyphone_distinguished_by_lexical_fields(self):
        payload = self.payload()
        result = next(
            item
            for item in payload["term_results"]
            if item["rank"] == 2)
        distinct = json.loads(json.dumps(
            result["additional_senses"][0],
            ensure_ascii=False))
        distinct["Pronunciation (English)"] = "different reading"
        distinct["Part of Speech (English)"] = "particle"
        distinct["Sentences"] = [
            "知五。",
            "知六。",
            "知七。",
            "知八。",
        ]
        distinct["Sentence Translations (English)"] = [
            "Fifth.",
            "Sixth.",
            "Seventh.",
            "Eighth.",
        ]
        result["additional_senses"].append(distinct)
        raw_text = json.dumps(payload, ensure_ascii=False)

        repaired = repair_compact_response(
            raw_text,
            self.pipeline,
            self.chunk)
        repaired_result = next(
            item
            for item in json.loads(
                repaired.candidate_raw_text)["term_results"]
            if item["rank"] == 2)
        report = inspect_pipeline_response(
            raw_text,
            self.pipeline,
            self.chunk,
            use_compact_source_results=True,
            use_local_example_emphasis=True)

        self.assertEqual(
            len(repaired_result["additional_senses"]),
            2)
        self.assertNotIn(
            "discard_duplicate_optional_sense",
            {change["rule_id"] for change in repaired.changes})
        self.assertNotIn(
            "duplicate_source_sense",
            {problem["code"] for problem in report["problems"]})

    def test_audit_artifacts_preserve_raw_and_verify_both_hashes(self):
        raw_text = json.dumps(self.payload(), ensure_ascii=False)
        result = repair_compact_response(
            raw_text,
            self.pipeline,
            self.chunk)
        value = ValidatedPipelineResponse(
            {"cards": []},
            local_repair_audit=result.audit_record(),
            effective_raw_text=result.candidate_raw_text)

        with TemporaryDirectory() as directory:
            attempt_path = Path(directory)
            (attempt_path / "raw.txt").write_text(
                raw_text,
                encoding="utf-8")

            GenerationJobStore._write_local_repair_artifacts(
                attempt_path,
                value)

            self.assertEqual(
                (attempt_path / "raw.txt").read_text(encoding="utf-8"),
                raw_text)
            self.assertEqual(
                (attempt_path / "repaired_raw.txt").read_text(
                    encoding="utf-8"),
                result.candidate_raw_text)
            audit = json.loads(
                (attempt_path / "local_repair.json").read_text(
                    encoding="utf-8"))
            self.assertEqual(
                audit["candidate_sha256"],
                result.candidate_sha256)

            (attempt_path / "raw.txt").write_text(
                raw_text + " ",
                encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError,
                    "hashes do not match"):
                GenerationJobStore._write_local_repair_artifacts(
                    attempt_path,
                    value)

            unchanged = repair_compact_response(
                result.candidate_raw_text,
                self.pipeline,
                self.chunk)
            self.assertFalse(unchanged.changed)
            (attempt_path / "raw.txt").write_text(
                unchanged.original_raw_text,
                encoding="utf-8")
            GenerationJobStore._write_local_repair_artifacts(
                attempt_path,
                ValidatedPipelineResponse(
                    {"cards": []},
                    local_repair_audit=unchanged.audit_record(),
                    effective_raw_text=unchanged.candidate_raw_text))
            self.assertFalse(
                (attempt_path / "repaired_raw.txt").exists())
            self.assertFalse(
                (attempt_path / "local_repair.json").exists())

    def test_invalid_repaired_response_persists_matching_audit_artifacts(self):
        payload = self.payload()
        payload["term_results"][0]["additional_senses"][0][
            "Sentences"][0] = "曉一。"
        raw_text = json.dumps(payload, ensure_ascii=False)
        validator = make_pipeline_response_validator(
            self.pipeline,
            use_compact_source_results=True,
            use_local_example_emphasis=True)

        with self.assertRaises(
                process_text.GeneratedCardValidationError) as captured:
            validator(raw_text, self.chunk)
        error = captured.exception

        with TemporaryDirectory() as directory:
            root = Path(directory)
            attempt_path = root / "attempt"
            attempt_path.mkdir()
            store = GenerationJobStore(root / "jobs")
            store.write_raw(attempt_path, raw_text)
            store.write_error(
                attempt_path,
                error,
                transient=False)

            repaired = (
                attempt_path / "repaired_raw.txt").read_text(
                    encoding="utf-8")
            audit = json.loads(
                (attempt_path / "local_repair.json").read_text(
                    encoding="utf-8"))
            self.assertEqual(
                audit["original_sha256"],
                error.local_repair_audit["original_sha256"])
            self.assertEqual(
                audit["candidate_sha256"],
                error.local_repair_audit["candidate_sha256"])
            self.assertEqual(
                audit["candidate_sha256"],
                hashlib.sha256(
                    repaired.encode("utf-8")).hexdigest())

    def test_v10_validation_ignores_provider_markup_and_emphasizes_locally(self):
        raw_text = json.dumps(
            self.payload(),
            ensure_ascii=False)

        report = inspect_pipeline_response(
            raw_text,
            self.pipeline,
            self.chunk,
            use_compact_source_results=True,
            use_local_example_emphasis=True)

        self.assertTrue(report["valid"], report["problems"])
        self.assertTrue(report["local_repair_applied"])
        cards = report["canonical_response"]["cards"]
        additional = next(
            card
            for card in cards
            if card["Translation (English)"] == "understand")
        self.assertEqual(
            additional["Sentences"].split("|"),
            [
                "<strong>知</strong>一。",
                "<strong>知</strong>二。",
                "<strong>知</strong>三。",
            ])

        validator = make_pipeline_response_validator(
            self.pipeline,
            use_compact_source_results=True,
            use_local_example_emphasis=True)
        validated = validator(raw_text, self.chunk)
        self.assertIsInstance(
            validated,
            ValidatedPipelineResponse)
        self.assertTrue(
            validated.local_repair_audit["changed"])

    def test_v10_accepts_inflection_when_literal_emphasis_cannot_be_added(self):
        pipeline = context_pipeline("french")
        request_chunk = chunk("courir")
        payload = {
            "term_results": [{
                "rank": 1,
                "contextual_sense": {
                    "Translation (English)": "run",
                    "Dictionary Meaning (English)": "To move quickly.",
                },
                "additional_senses": [{
                    "Translation (English)": "race",
                    "Dictionary Meaning (English)": (
                        "To take part in a running race."),
                    "Sentences": [
                        "Elle court vite.",
                        "Nous courons ensemble.",
                        "Ils couraient hier.",
                    ],
                    "Sentence Translations (English)": [
                        "She runs quickly.",
                        "We run together.",
                        "They were running yesterday.",
                    ],
                }],
            }],
            "source_context_translations": [{
                "context_id": "context-1",
                "translation": "An example with the verb to run.",
            }],
        }

        report = inspect_pipeline_response(
            json.dumps(payload, ensure_ascii=False),
            pipeline,
            request_chunk,
            use_compact_source_results=True,
            use_local_example_emphasis=True)

        self.assertTrue(report["valid"], report["problems"])
        additional = report["canonical_response"]["cards"][1]
        self.assertNotIn("<strong>", additional["Sentences"])

    def test_v10_rejects_missing_complete_term_in_classical_chinese(self):
        payload = self.payload()
        payload["term_results"][0]["additional_senses"][0][
            "Sentences"][0] = "曉一。"

        report = inspect_pipeline_response(
            json.dumps(payload, ensure_ascii=False),
            self.pipeline,
            self.chunk,
            use_compact_source_results=True,
            use_local_example_emphasis=True)

        problems = [
            problem
            for problem in report["problems"]
            if problem["code"] == "missing_exact_form_source_term"
        ]
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["source_rank"], 2)
        self.assertEqual(
            problems[0]["repair_target"],
            {
                "kind": "source_rank",
                "source_rank": 2,
            })

    def test_v10_accepts_and_emphasizes_repeated_classical_chinese_term(self):
        payload = self.payload()
        payload["term_results"][0]["additional_senses"][0][
            "Sentences"][0] = "知而復知。"

        report = inspect_pipeline_response(
            json.dumps(payload, ensure_ascii=False),
            self.pipeline,
            self.chunk,
            use_compact_source_results=True,
            use_local_example_emphasis=True)

        self.assertTrue(report["valid"], report["problems"])
        additional = next(
            card
            for card in report["canonical_response"]["cards"]
            if card["Translation (English)"] == "understand")
        self.assertEqual(
            additional["Sentences"].split("|")[0],
            "<strong>知</strong>而復<strong>知</strong>。")


if __name__ == "__main__":
    unittest.main()
