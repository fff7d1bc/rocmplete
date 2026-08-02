import hashlib
import stat
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

from tools.archive_probe import probe


class ArchiveProbeTests(unittest.TestCase):
    def test_probe_hashes_regular_members_without_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pack.zip"
            with zipfile.ZipFile(str(path), "w") as archive:
                archive.writestr("b/workflow.json", b"workflow")
                archive.writestr("a/model.bin", b"model")

            result = probe(path)

            self.assertEqual(result["size"], path.stat().st_size)
            self.assertEqual(
                result["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                [item["name"] for item in result["members"]],
                ["a/model.bin", "b/workflow.json"],
            )
            self.assertEqual(
                result["members"][0]["sha256"],
                hashlib.sha256(b"model").hexdigest(),
            )
            self.assertFalse((Path(directory) / "a/model.bin").exists())

    def test_probe_flags_unsafe_and_symlink_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pack.zip"
            link = zipfile.ZipInfo("link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(str(path), "w") as archive:
                archive.writestr("../escape", b"bad")
                archive.writestr(link, "target")

            result = probe(path)
            by_name = {item["name"]: item for item in result["members"]}

            self.assertFalse(by_name["../escape"]["safe_path"])
            self.assertEqual(by_name["link"]["type"], "symlink")
            self.assertIsNone(by_name["link"]["sha256"])

    def test_member_selection_requires_one_exact_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pack.zip"
            with zipfile.ZipFile(str(path), "w") as archive:
                archive.writestr("workflow.json", b"workflow")

            result = probe(path, ("workflow.json",))
            self.assertEqual(len(result["members"]), 1)
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                probe(path, ("missing.json",))

    def test_duplicate_member_names_are_reported_and_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pack.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(str(path), "w") as archive:
                    archive.writestr("workflow.json", b"first")
                    archive.writestr("workflow.json", b"second")

            result = probe(path)
            self.assertEqual(result["duplicate_names"], ["workflow.json"])
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                probe(path, ("workflow.json",))


if __name__ == "__main__":
    unittest.main()
