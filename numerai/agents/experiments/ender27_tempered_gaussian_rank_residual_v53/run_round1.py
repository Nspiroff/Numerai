"""Launch one exact Ender27 Round-1 config through the frozen bootstrap."""

from __future__ import annotations

import sys


def _require_outer_launch() -> str:
    option = getattr(sys, "_xoptions", {}).get("pycache_prefix")
    if (
        sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.dont_write_bytecode is not True
        or sys.flags.safe_path != 1
        or type(option) is not str
        or not option
        or sys.pycache_prefix != option
    ):
        raise ValueError(
            "Ender27 launcher requires isolated Python -I -B -P and an exact "
            "nonempty -X pycache_prefix."
        )
    return option


def main() -> int:
    option = _require_outer_launch()

    # Do not import filesystem or process modules until isolation is proven.
    from pathlib import Path
    import subprocess

    bootstrap = Path(__file__).with_name("training_bootstrap.py")
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-P",
            "-X",
            f"pycache_prefix={option}",
            str(bootstrap),
            "1",
            *sys.argv[1:],
        ],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
