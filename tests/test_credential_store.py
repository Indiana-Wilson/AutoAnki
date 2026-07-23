import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import credential_store


class CredentialStoreTests(unittest.TestCase):
    def test_save_load_and_delete_api_key(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_directory = Path(temporary_directory) / "autoanki"
            with patch.dict(
                    os.environ,
                    {"AUTOANKI_CONFIG_DIR": str(config_directory)}):
                credential_store.save_api_key("  test-api-key  ")

                self.assertEqual(
                    credential_store.load_api_key(),
                    "test-api-key")
                self.assertTrue(credential_store.delete_api_key())
                self.assertIsNone(credential_store.load_api_key())
                self.assertFalse(credential_store.delete_api_key())

    def test_saved_key_uses_owner_only_permissions_on_posix(self):
        if os.name != "posix":
            self.skipTest("POSIX permissions are not available")

        with tempfile.TemporaryDirectory() as temporary_directory:
            config_directory = Path(temporary_directory) / "autoanki"
            with patch.dict(
                    os.environ,
                    {"AUTOANKI_CONFIG_DIR": str(config_directory)}):
                credential_store.save_api_key("test-api-key")

                directory_mode = stat.S_IMODE(config_directory.stat().st_mode)
                file_mode = stat.S_IMODE(
                    credential_store.get_api_key_path().stat().st_mode)

        self.assertEqual(directory_mode, 0o700)
        self.assertEqual(file_mode, 0o600)

    def test_empty_api_key_is_rejected_without_creating_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_directory = Path(temporary_directory) / "autoanki"
            with (
                    patch.dict(
                        os.environ,
                        {"AUTOANKI_CONFIG_DIR": str(config_directory)}),
                    self.assertRaises(ValueError)):
                credential_store.save_api_key("   ")

            self.assertFalse(
                (config_directory / "openai_api_key").exists())


if __name__ == "__main__":
    unittest.main()
