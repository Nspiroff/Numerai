"""Launch one exact Ender22 Round-1 config through the frozen bootstrap."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    if type(sys.pycache_prefix) is not str or not sys.pycache_prefix:
        raise ValueError("Ender22 launcher requires a frozen pycache prefix.")
    bootstrap = Path(__file__).with_name("training_bootstrap.py")
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-X",
            f"pycache_prefix={sys.pycache_prefix}",
            str(bootstrap),
            "1",
            *sys.argv[1:],
        ],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
