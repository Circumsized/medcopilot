"""W5.7: the compliance notice is one string, shown on the first screen and appended to answers.

The finding this pins: the UI previously had no persistent disclaimer ("AI 不能
替代医师" appeared only at the end of an answer, and only for answers the API
actually produced). Two properties matter:

1. **同源**: the banner and the appended notice render the same constant, so they
   cannot drift apart.
2. **工件一致**: that constant is what the policy artifact mirrors as
   ``disclaimer_base``; extracting it into a named constant must not change a
   single byte, or ``policy_check`` would go red for the wrong reason.
"""

from __future__ import annotations

import re
from pathlib import Path

from medcopilot.workflow.safety_guard import (
    DISCLAIMER_BANNER,
    DISCLAIMER_TEXT,
    NO_EVIDENCE_PREFIX,
    MedicalSafetyGuard,
)

APP_SOURCE = Path(__file__).resolve().parents[1] / "ui" / "app.py"


def test_appended_notice_is_the_shared_constant():
    assert MedicalSafetyGuard.post_process_response("答复正文") == "答复正文" + DISCLAIMER_TEXT
    assert MedicalSafetyGuard.post_process_response("答复正文", has_retrieved_evidence=False) == (
        "答复正文" + NO_EVIDENCE_PREFIX + DISCLAIMER_TEXT
    )


def test_banner_reuses_the_same_text_and_carries_the_emergency_pointer():
    # The banner must contain the shared notice (whitespace-normalised: the banner
    # inlines it on one line instead of after a horizontal rule).
    normalised_banner = re.sub(r"\s+", "", DISCLAIMER_BANNER)
    normalised_notice = re.sub(r"\s+", "", DISCLAIMER_TEXT.replace("\n\n---\n", ""))
    assert normalised_notice in normalised_banner
    assert "不能替代执业医师" in DISCLAIMER_BANNER
    assert "120" in DISCLAIMER_BANNER


def test_banner_is_rendered_on_the_first_screen_of_the_ui():
    source = APP_SOURCE.read_text(encoding="utf-8")
    assert "DISCLAIMER_BANNER" in source
    # A banner that is imported but never rendered would satisfy an import check
    # while leaving the first screen unchanged.
    assert re.search(r"gr\.Markdown\(\s*DISCLAIMER_BANNER\s*\)", source)


def test_notice_text_still_matches_the_policy_artifact_mirror():
    from medcopilot.safety.policy import current_code_content

    content = current_code_content()
    assert content["disclaimer_base"] == DISCLAIMER_TEXT
    assert content["disclaimer_no_evidence_prefix"] == NO_EVIDENCE_PREFIX
