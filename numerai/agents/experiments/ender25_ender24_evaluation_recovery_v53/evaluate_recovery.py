"""Stdlib-only launcher for the sealed Ender25 evaluation recovery."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    option = getattr(sys, "_xoptions", {}).get("pycache_prefix")
    if (
        sys.flags.isolated != 1
        or sys.flags.safe_path != 1
        or sys.flags.dont_write_bytecode != 1
        or not sys.dont_write_bytecode
    ):
        raise ValueError("Ender25 recovery launcher requires Python -I -B -P.")
    if type(option) is not str or not option or sys.pycache_prefix != option:
        raise ValueError(
            "Ender25 recovery launcher requires an exact -X pycache_prefix."
        )
    bootstrap = Path(__file__).with_name("evaluation_bootstrap.py")
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-P",
            "-X",
            f"pycache_prefix={option}",
            str(bootstrap),
            *sys.argv[1:],
        ],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
