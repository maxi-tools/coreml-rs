from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
ACTIONLINT_CONFIG = ROOT / ".github" / "actionlint.yaml"
PINNED_ACTION = re.compile(r"uses:\s*[^\s@]+/[^^\s@]+@[0-9a-f]{40}(?:\s|$)")
USES_ACTION = re.compile(r"uses:\s*[^\s@]+/[^\s@]+@[^\s]+")


class WorkflowPolicyTests(unittest.TestCase):
    def test_pull_request_routing_uses_isolated_fork_lane(self) -> None:
        text = CI_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("rust:", text)
        self.assertIn("rust-fork:", text)
        self.assertIn("github.event.pull_request.head.repo.full_name == github.repository", text)
        self.assertIn("github.event.pull_request.head.repo.full_name != github.repository", text)
        self.assertIn("runs-on: warp-macos-15-arm64-6x", text)
        self.assertIn("runs-on: macos-latest", text)
        self.assertNotIn("runs-on: [self-hosted, macOS, ARM64]", text)

    def test_workflow_does_not_persist_org_tokens(self) -> None:
        text = CI_WORKFLOW.read_text(encoding="utf-8")

        self.assertNotIn("APP_PRIVATE_KEY", text)
        self.assertNotIn("actions/create-github-app-token", text)
        self.assertNotIn("git config --global", text)

    def test_actionlint_knows_custom_warp_label(self) -> None:
        text = ACTIONLINT_CONFIG.read_text(encoding="utf-8")

        self.assertIn("warp-macos-15-arm64-6x", text)

    def test_third_party_actions_are_pinned_to_shas(self) -> None:
        text = CI_WORKFLOW.read_text(encoding="utf-8")
        unpinned = [line.strip() for line in text.splitlines() if USES_ACTION.search(line) and not PINNED_ACTION.search(line)]

        self.assertEqual([], unpinned)


if __name__ == "__main__":
    unittest.main()
