"""PaddleOCR availability probe using the OCRTool contract.

The probe never invents OCR text.  Missing image/dependency is reported as
``not-configured`` with exit code 2; inference failures are reported honestly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medcopilot.agents.ocr_tool import OCRTool

REPORT_SCHEMA_VERSION = 1


def probe(image: str | None) -> dict[str, Any]:
    """Run ``OCRTool.extract_text_with_status`` and return redacted metadata."""
    result: dict[str, Any] = {"report_schema_version": REPORT_SCHEMA_VERSION}
    if not image:
        result.update({"status": "not-configured", "reason": "--image is required for an OCR probe"})
        return result
    image_path = Path(image)
    if not image_path.is_file():
        result.update({"status": "not-configured", "reason": "image path is missing or unreadable"})
        return result

    lines, tool_status = OCRTool().extract_text_with_status(str(image_path))
    result["ocr_status"] = tool_status
    result["line_count"] = len(lines)
    if tool_status == "ok":
        result["status"] = "ok"
    elif tool_status == "engine_unavailable":
        result.update({"status": "not-configured", "reason": "PaddleOCR optional dependency is unavailable"})
    else:
        result["status"] = "error"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe OCRTool and optional PaddleOCR availability")
    parser.add_argument("--image", default=None, help="path to a local image (never uploaded)")
    parser.add_argument("--output", default="outputs/ocr_probe.json")
    args = parser.parse_args(argv)

    report = probe(args.image)
    output = Path(args.output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        print(json.dumps({"status": "error", "reason": "report write failed"}))
        return 1
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] == "not-configured":
        return 2
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
