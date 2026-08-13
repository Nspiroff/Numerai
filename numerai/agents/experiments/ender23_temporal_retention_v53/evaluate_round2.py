"""Stdlib-only launcher for the frozen Ender23 Round-2 evaluator."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    if type(sys.pycache_prefix) is not str or not sys.pycache_prefix:
        raise ValueError("Ender23 evaluator launcher requires a frozen pycache prefix.")
    bootstrap = Path(__file__).with_name("training_bootstrap.py")
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-P",
            "-X",
            f"pycache_prefix={sys.pycache_prefix}",
            str(bootstrap),
            "evaluate",
            "2",
            *sys.argv[1:],
        ],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
