import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gui_preferences


class GuiPreferenceStoreTests(unittest.TestCase):
    def test_missing_file_loads_independent_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            first = gui_preferences.load_preferences(path)
            second = gui_preferences.load_preferences(path)

        self.assertEqual(first, gui_preferences.DEFAULTS)
        first["source_card_directions"].append("word_to_meaning")
        self.assertNotEqual(first, second)

    def test_preferences_round_trip_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gui.json"
            preferences = gui_preferences.default_preferences()
            preferences.update({
                "source_key": "custom-source",
                "source_language_overrides": {
                    "custom-source": "classical_chinese_warring_states",
                },
                "source_card_directions": [
                    "context",
                    "word_to_meaning",
                ],
                "source_include_context_nuance": True,
                "source_file_use_gpu": False,
                "main_tab": "advanced",
            })

            saved = gui_preferences.save_preferences(preferences, path)
            loaded = gui_preferences.load_preferences(path)

            self.assertEqual(saved, loaded)
            self.assertEqual(loaded["source_key"], "custom-source")
            self.assertTrue(loaded["source_include_context_nuance"])
            self.assertFalse(loaded["source_file_use_gpu"])
            self.assertEqual(loaded["main_tab"], "advanced")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["version"],
                gui_preferences.CONFIG_VERSION)

    def test_unknown_future_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gui.json"
            value = {
                "version": gui_preferences.CONFIG_VERSION,
                "preferences": {
                    **gui_preferences.default_preferences(),
                    "future_setting": "safe to ignore",
                },
            }
            path.write_text(json.dumps(value), encoding="utf-8")

            loaded = gui_preferences.load_preferences(path)

        self.assertNotIn("future_setting", loaded)

    def test_safeguard_authorizations_are_not_in_schema(self):
        self.assertNotIn(
            "source_paid_authorized",
            gui_preferences.DEFAULTS)
        self.assertNotIn(
            "source_codex_authorized",
            gui_preferences.DEFAULTS)
        self.assertNotIn(
            "prompt_editing_enabled",
            gui_preferences.DEFAULTS)

    def test_wrong_type_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gui.json"
            path.write_text(
                json.dumps({
                    "version": gui_preferences.CONFIG_VERSION,
                    "preferences": {
                        **gui_preferences.default_preferences(),
                        "source_automatic_repair": "yes",
                    },
                }),
                encoding="utf-8")

            with self.assertRaisesRegex(
                    ValueError,
                    "source_automatic_repair"):
                gui_preferences.load_preferences(path)


if __name__ == "__main__":
    unittest.main()
