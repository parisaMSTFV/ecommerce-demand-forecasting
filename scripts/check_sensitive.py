"""Fail CI if common credential or private-infrastructure patterns are tracked."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
TEXT_SUFFIXES = {
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

PATTERNS = {
    "private key": re.compile("BEGIN " + "PRIVATE KEY"),
    "cloud access key": re.compile("AK" + "IA[0-9A-Z]{16}"),
    "GitHub token": re.compile("gh" + "p_[A-Za-z0-9]{30,}"),
    "API token": re.compile("sk" + "-[A-Za-z0-9]{20,}"),
    "password assignment": re.compile(r"(?i)password\s*[:=]\s*[\"'][^\"']+[\"']"),
    "database connection": re.compile(r"(?i)(server|host|database)\s*=\s*[^;\s]+;"),
}


def candidate_files() -> list[Path]:
    try:
        output = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        paths = [ROOT / line for line in output.splitlines()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        paths = list(ROOT.rglob("*"))
    return [
        path
        for path in paths
        if path.is_file()
        and path.resolve() != SELF
        and path.suffix.lower() in TEXT_SUFFIXES
        and ".git" not in path.parts
        and ".venv" not in path.parts
    ]


def main() -> None:
    findings: list[str] = []
    for path in candidate_files():
        content = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append(f"{path.relative_to(ROOT)}: {label}")
    if findings:
        raise SystemExit("Sensitive-content check failed:\n" + "\n".join(findings))
    print(f"Sensitive-content check passed for {len(candidate_files())} text files.")


if __name__ == "__main__":
    main()
