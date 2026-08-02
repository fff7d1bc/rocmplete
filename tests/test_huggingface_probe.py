import io
import unittest
from unittest.mock import patch

from tools.huggingface_probe import (
    _request_json,
    probe_file,
    probe_repository,
    probe_revision,
)


class _Response:
    def __init__(self, contents):
        self.contents = contents

    def __enter__(self):
        return io.BytesIO(self.contents)

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class HuggingFaceProbeTests(unittest.TestCase):
    @patch("tools.huggingface_probe.urllib.request.urlopen")
    def test_request_uses_bearer_token_without_putting_it_in_url(
        self, urlopen
    ):
        urlopen.return_value = _Response(b'{"id": "owner/model"}')

        self.assertEqual(
            _request_json("models/owner/model", "secret"),
            {"id": "owner/model"},
        )
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://huggingface.co/api/models/owner/model",
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")

    @patch("tools.huggingface_probe._request_json")
    def test_revision_sorts_files_and_keeps_lfs_identity(self, request_json):
        request_json.return_value = {
            "id": "owner/model",
            "sha": "a" * 40,
            "private": False,
            "gated": False,
            "cardData": {"license": "apache-2.0"},
            "siblings": [
                {"rfilename": "z.txt", "size": 4, "blobId": "git"},
                {
                    "rfilename": "a.bin",
                    "lfs": {"size": 10, "sha256": "b" * 64},
                },
            ],
        }

        result = probe_revision("owner/model", "a" * 40)

        self.assertEqual(
            [item["path"] for item in result["files"]],
            ["a.bin", "z.txt"],
        )
        self.assertEqual(result["files"][0]["sha256"], "b" * 64)
        self.assertIsNone(result["files"][1]["sha256"])

    @patch("tools.huggingface_probe._request_json")
    def test_file_requires_one_exact_path(self, request_json):
        request_json.return_value = {
            "id": "owner/model",
            "sha": "a" * 40,
            "siblings": [{"rfilename": "model.bin", "size": 7}],
        }

        result = probe_file("owner/model", "a" * 40, "model.bin")

        self.assertEqual(result["file"]["path"], "model.bin")
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            probe_file("owner/model", "a" * 40, "missing.bin")

    @patch("tools.huggingface_probe._request_json")
    def test_repository_omits_large_file_inventory(self, request_json):
        request_json.return_value = {
            "id": "owner/model",
            "sha": "a" * 40,
            "siblings": [{"rfilename": "model.bin"}],
        }

        self.assertNotIn("files", probe_repository("owner/model"))


if __name__ == "__main__":
    unittest.main()
