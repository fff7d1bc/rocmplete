import re
import unittest
from pathlib import Path


class ContinuousIntegrationTests(unittest.TestCase):
    def test_workflow_runs_hosted_source_checks_without_publishing_images(self):
        workflow = (
            Path(__file__).parents[1] / ".github" / "workflows" / "checks.yml"
        ).read_text()
        jobs = workflow.split("jobs:\n", 1)[1]

        self.assertEqual(
            re.findall(r"^  ([a-z0-9-]+):$", jobs, flags=re.MULTILINE),
            ["tier-1"],
        )
        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertIn("runs-on: ubuntu-24.04", jobs)
        self.assertIn("persist-credentials: false", jobs)
        self.assertNotIn("self-hosted", workflow)
        self.assertNotIn("packages: write", workflow)
        self.assertNotIn("podman push", workflow)
        self.assertNotIn("docker push", workflow)
        self.assertNotIn("upload-artifact", workflow)


if __name__ == "__main__":
    unittest.main()
