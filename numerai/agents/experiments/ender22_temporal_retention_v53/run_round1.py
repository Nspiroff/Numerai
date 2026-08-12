"""Launch one exact Ender22 Round-1 config through the frozen bootstrap."""

import os
from pathlib import Path
import sys


bootstrap = Path(__file__).with_name("training_bootstrap.py")
os.execv(
    sys.executable,
    [sys.executable, "-I", "-B", "-X", f"pycache_prefix={sys.pycache_prefix}", str(bootstrap), "1", *sys.argv[1:]],
)
