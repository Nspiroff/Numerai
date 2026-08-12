"""Stdlib-only launcher for the frozen Ender22 Round-1 evaluator."""

from __future__ import annotations

import os
from pathlib import Path
import sys


bootstrap = Path(__file__).with_name("training_bootstrap.py")
os.execv(
    sys.executable,
    [
        sys.executable,
        "-I",
        "-B",
        "-P",
        "-X",
        f"pycache_prefix={sys.pycache_prefix}",
        str(bootstrap),
        "evaluate",
        "1",
        *sys.argv[1:],
    ],
)
