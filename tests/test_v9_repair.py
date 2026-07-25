import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import process_text
from source_generation.models import (
    ContextUnit,
    GenerationChunk,
    GenerationWord,
)
from source_generation.repair import (
    build_compact_repair_chunk,
    derive_compact_repair_scope,
    merge_compact_repair,
)


def sample_chunk():
    context_a = ContextUnit(
        context_id="ctx-a",
        mode="sentence",
        start_offset=0,
        end_offset=4,
        text="道可道。",
        section_ids=("s1",),
        sentence_ids=("x1",),
        word_ranks=(1, 2),
    )
    context_b = ContextUnit(
        context_id="ctx-b",
        mode="sentence",
        start_offset=4,
        end_offset=8,
        text="名可名。",
        section_ids=("s1",),
        sentence_ids=("x2",),
        word_ranks=(3,),
    )
    words = (
        GenerationWord(1, "道", "道", "s1", "x1", "ctx-a"),
        GenerationWord(2, "可", "可", "s1", "x1", "ctx-a"),
        GenerationWord(3, "名", "名", "s1", "x2", "ctx-b"),
    )
    return GenerationChunk(
        chunk_id="chunk-1",
        index=1,
        total=1,
        start_rank=1,
        end_rank=3,
        words=words,
        contexts=(context_a, context_b),
    )


def term(rank, value):
    return {
        process_text.SOURCE_RANK_FIELD_NAME: rank,
        process_text.SOURCE_CONTEXTUAL_SENSE_KEY: {"value": value},
        process_text.SOURCE_ADDITIONAL_SENSES_KEY: [],
    }


def context(context_id, value):
    return {
        process_text.SOURCE_CONTEXT_ID_FIELD_NAME: context_id,
        process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME: value,
    }


class CompactRepairTests(unittest.TestCase):
    def test_scope_expands_context_target_but_replaces_only_failed_parts(self):
        chunk = sample_chunk()
        report = {
            "problems": [
                {
                    "repair_target": {
                        "kind": "source_rank",
                        "source_rank": 3,
                    },
                },
                {
                    "repair_target": {
                        "kind": "context",
                        "context_id": "ctx-a",
                    },
                },
            ],
        }
        scope = derive_compact_repair_scope(report, chunk)
        self.assertEqual(scope.replace_ranks, (3,))
        self.assertEqual(scope.replace_context_ids, ("ctx-a",))
        self.assertEqual(scope.request_ranks, (1, 2, 3))
        self.assertEqual(scope.request_context_ids, ("ctx-a", "ctx-b"))
        repair_chunk = build_compact_repair_chunk(chunk, scope)
        self.assertEqual(
            tuple(word.rank for word in repair_chunk.words),
            (1, 2, 3))

    def test_unscoped_or_full_retry_problem_disables_selective_repair(self):
        chunk = sample_chunk()
        self.assertIsNone(derive_compact_repair_scope(
            {"problems": [{"code": "unknown"}]},
            chunk))
        self.assertIsNone(derive_compact_repair_scope(
            {
                "problems": [{
                    "full_retry_required": True,
                    "repair_target": {
                        "kind": "source_rank",
                        "source_rank": 1,
                    },
                }],
            },
            chunk))

    def test_merge_repairs_missing_rank_and_one_context_only(self):
        chunk = sample_chunk()
        scope = derive_compact_repair_scope(
            {
                "problems": [
                    {
                        "repair_target": {
                            "kind": "source_rank",
                            "source_rank": 2,
                        },
                    },
                    {
                        "repair_target": {
                            "kind": "context",
                            "context_id": "ctx-b",
                        },
                    },
                ],
            },
            chunk)
        base = {
            process_text.SOURCE_TERM_RESULTS_KEY: [
                term(1, "old-1"),
                term(3, "old-3"),
            ],
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: [
                context("ctx-a", "old-a"),
                context("ctx-b", "old-b"),
            ],
        }
        repair = {
            process_text.SOURCE_TERM_RESULTS_KEY: [
                term(2, "new-2"),
                term(3, "discarded-new-3"),
            ],
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: [
                context("ctx-a", "discarded-new-a"),
                context("ctx-b", "new-b"),
            ],
        }
        merged = json.loads(merge_compact_repair(
            json.dumps(base),
            json.dumps(repair),
            chunk,
            scope))
        self.assertEqual(
            [
                item["contextual_sense"]["value"]
                for item in merged[
                    process_text.SOURCE_TERM_RESULTS_KEY]
            ],
            ["old-1", "new-2", "old-3"])
        self.assertEqual(
            [
                item["translation"]
                for item in merged[
                    process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY]
            ],
            ["old-a", "new-b"])

    def test_merge_rejects_unknown_identity(self):
        chunk = sample_chunk()
        scope = derive_compact_repair_scope(
            {
                "problems": [{
                    "repair_target": {
                        "kind": "source_rank",
                        "source_rank": 1,
                    },
                }],
            },
            chunk)
        base = {
            process_text.SOURCE_TERM_RESULTS_KEY: [
                term(1, "a"),
                term(2, "b"),
                term(3, "c"),
                term(99, "unknown"),
            ],
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: [
                context("ctx-a", "a"),
                context("ctx-b", "b"),
            ],
        }
        repair = {
            process_text.SOURCE_TERM_RESULTS_KEY: [term(1, "new")],
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: [
                context("ctx-a", "a"),
            ],
        }
        with self.assertRaisesRegex(ValueError, "unknown identity"):
            merge_compact_repair(
                json.dumps(base),
                json.dumps(repair),
                chunk,
                scope)


if __name__ == "__main__":
    unittest.main()
