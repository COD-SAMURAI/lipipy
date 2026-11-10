"""Fail if detect-secrets finds a likely secret in any tracked file.

Jupyter cell ids (32 random hex characters) are excluded because they look like
high-entropy strings but are not secrets. Usage: ``python scripts/secret_scan.py``.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 - runs fixed commands only
import sys

EXCLUDE_LINES = r'"id": "[0-9a-f]{32}"'


def main() -> int:
    files = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    scan = subprocess.run(  # nosec B603 - fixed argv built from git output, no shell
        [sys.executable, "-m", "detect_secrets", "scan", "--exclude-lines", EXCLUDE_LINES, *files],
        capture_output=True,
        text=True,
        check=True,
    )
    results = json.loads(scan.stdout)["results"]
    if not results:
        print(f"No secrets found in {len(files)} tracked files.")
        return 0
    for path, findings in results.items():
        for f in findings:
            print(f"{path}:{f['line_number']}: {f['type']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
