import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import runtime_paths


class RuntimePathTests(unittest.TestCase):
    def test_source_checkout_uses_project_prompts_and_output(self):
        with patch.object(runtime_paths, "is_frozen", return_value=False):
            self.assertEqual(
                runtime_paths.get_prompt_directory(),
                PROJECT_ROOT / "input" / "prompts")
            self.assertEqual(
                runtime_paths.get_output_directory(),
                PROJECT_ROOT / "output")
            self.assertEqual(
                runtime_paths.get_corpus_output_directory(),
                PROJECT_ROOT / "output" / "corpora")

    def test_frozen_prompts_are_copied_once_and_edits_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            bundle_root = temporary_root / "bundle"
            bundled_prompts = bundle_root / "input" / "prompts"
            bundled_prompts.mkdir(parents=True)
            (bundled_prompts / "english_vocab").write_text(
                "bundled prompt",
                encoding="utf-8")
            config_root = temporary_root / "config"

            with (
                    patch.dict(
                        os.environ,
                        {"AUTOANKI_CONFIG_DIR": str(config_root)}),
                    patch.object(
                        runtime_paths,
                        "is_frozen",
                        return_value=True),
                    patch.object(
                        runtime_paths,
                        "get_resource_root",
                        return_value=bundle_root)):
                prompt_directory = runtime_paths.get_prompt_directory()
                editable_prompt = prompt_directory / "english_vocab"
                self.assertEqual(
                    editable_prompt.read_text(encoding="utf-8"),
                    "bundled prompt")

                editable_prompt.write_text(
                    "user edit",
                    encoding="utf-8")
                (bundled_prompts / "english_vocab").write_text(
                    "new bundled default",
                    encoding="utf-8")
                (bundled_prompts / "french_vocab").write_text(
                    "new prompt",
                    encoding="utf-8")

                runtime_paths.get_prompt_directory()

                self.assertEqual(
                    editable_prompt.read_text(encoding="utf-8"),
                    "user edit")
                self.assertEqual(
                    (prompt_directory / "french_vocab").read_text(
                        encoding="utf-8"),
                    "new prompt")

    def test_frozen_outputs_are_outside_temporary_bundle(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_root = Path(temporary_directory) / "config"
            with (
                    patch.dict(
                        os.environ,
                        {"AUTOANKI_CONFIG_DIR": str(config_root)}),
                    patch.object(
                        runtime_paths,
                        "is_frozen",
                        return_value=True)):
                self.assertEqual(
                    runtime_paths.get_output_directory(),
                    config_root / "output")
                self.assertEqual(
                    runtime_paths.get_corpus_output_directory(),
                    config_root / "output" / "corpora")


if __name__ == "__main__":
    unittest.main()
