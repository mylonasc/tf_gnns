#!/usr/bin/env python3
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def badge_color(coverage_pct: int) -> str:
    if coverage_pct >= 90:
        return "brightgreen"
    if coverage_pct >= 80:
        return "green"
    if coverage_pct >= 70:
        return "yellowgreen"
    if coverage_pct >= 60:
        return "yellow"
    if coverage_pct >= 50:
        return "orange"
    return "red"


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: update_coverage_badge.py <coverage.xml> <output.json>", file=sys.stderr)
        return 1

    coverage_xml = Path(sys.argv[1])
    out_json = Path(sys.argv[2])

    if not coverage_xml.exists():
        print(f"Coverage file not found: {coverage_xml}", file=sys.stderr)
        return 1

    root = ET.parse(coverage_xml).getroot()
    line_rate_raw = root.attrib.get("line-rate")
    if line_rate_raw is None:
        print("Coverage XML does not contain line-rate", file=sys.stderr)
        return 1

    coverage_pct = int(round(float(line_rate_raw) * 100))
    payload = {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{coverage_pct}%",
        "color": badge_color(coverage_pct),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Coverage: {coverage_pct}%")
    print(f"Wrote badge data: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
