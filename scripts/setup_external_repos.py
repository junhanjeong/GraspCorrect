#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graspcorrect.baselines.registry import BASELINE_REPOS


def clone_or_report(name: str, url: str, root: Path) -> None:
    dest = root / name
    if dest.exists():
        print(f"[skip] {name}: {dest} already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[clone] {name}: {url} -> {dest}")
    subprocess.run(["git", "clone", url, str(dest)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone official benchmark and baseline repositories.")
    parser.add_argument("--root", default="external", help="Directory where external repos are cloned.")
    parser.add_argument("--all", action="store_true", help="Clone every known external repository.")
    parser.add_argument("--repo", action="append", choices=sorted(BASELINE_REPOS), help="Repository key to clone.")
    args = parser.parse_args()

    selected = sorted(BASELINE_REPOS) if args.all else (args.repo or [])
    if not selected:
        parser.error("Pass --all or at least one --repo.")
    root = Path(args.root)
    for name in selected:
        clone_or_report(name, BASELINE_REPOS[name], root)


if __name__ == "__main__":
    main()
