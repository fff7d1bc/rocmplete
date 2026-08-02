import io
import unittest
from unittest.mock import patch

from tools.civitai_probe import _request_json, probe, probe_hash, search


class _Response:
    def __init__(self, contents):
        self.contents = contents

    def __enter__(self):
        return io.BytesIO(self.contents)

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class CivitaiProbeTests(unittest.TestCase):
    @patch("tools.civitai_probe.urllib.request.urlopen")
    def test_request_uses_bearer_token_without_putting_it_in_url(
        self, urlopen
    ):
        urlopen.return_value = _Response(b'{"id": 7}')

        self.assertEqual(_request_json("models/7", "secret"), {"id": 7})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://civitai.com/api/v1/models/7")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")

    @patch("tools.civitai_probe._request_json")
    def test_version_summary_keeps_catalog_fields(self, request_json):
        request_json.return_value = {
            "id": 22,
            "modelId": 11,
            "name": "Illustrious",
            "baseModel": "Illustrious",
            "files": [
                {
                    "id": 33,
                    "name": "workflow.json",
                    "type": "Training Data",
                    "sizeKB": 12.5,
                    "hashes": {"SHA256": "abc"},
                }
            ],
            "images": [{"id": 44, "type": "image", "meta": {"seed": 1}}],
        }

        result = probe("version", 22, token="secret")

        self.assertEqual(result["id"], 22)
        self.assertEqual(result["files"][0]["name"], "workflow.json")
        self.assertEqual(result["files"][0]["hashes"]["SHA256"], "abc")
        self.assertEqual(result["images"][0]["meta"], {"seed": 1})

    @patch("tools.civitai_probe._request_json")
    def test_search_encodes_query_and_summarizes_versions(self, request_json):
        request_json.return_value = {
            "items": [
                {
                    "id": 1,
                    "name": "A model",
                    "type": "Checkpoint",
                    "creator": {"username": "author"},
                    "modelVersions": [{"id": 2, "name": "v1", "files": []}],
                }
            ]
        }

        result = search("name with spaces", token="secret")

        self.assertEqual(result["items"][0]["versions"][0]["id"], 2)
        self.assertEqual(
            request_json.call_args.args[0],
            "models?query=name+with+spaces&limit=20",
        )

    @patch("tools.civitai_probe._request_json")
    def test_hash_lookup_uses_hash_endpoint(self, request_json):
        request_json.return_value = {"id": 4, "files": [], "images": []}

        self.assertEqual(probe_hash("ABC DEF")["id"], 4)
        self.assertEqual(
            request_json.call_args.args[0],
            "model-versions/by-hash/ABC%20DEF",
        )


if __name__ == "__main__":
    unittest.main()
