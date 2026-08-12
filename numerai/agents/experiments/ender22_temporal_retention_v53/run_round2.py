"""Launch one exact Ender22 Round-2 config through the frozen bootstrap."""

import os
from pathlib import Path
import sys


bootstrap = Path(__file__).with_name("training_bootstrap.py")
os.execv(
    sys.executable,
    [sys.executable, "-I", "-B", "-X", f"pycache_prefix={sys.pycache_prefix}", str(bootstrap), "2", *sys.argv[1:]],
)
