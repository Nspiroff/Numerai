from pathlib import Path
import runpy


MAKE_CONFIG = runpy.run_path(str(Path(__file__).with_name("base_d8.py")))[
    "make_config"
]
CONFIG = MAKE_CONFIG("target_teager2b_20", "r1_teager2b_d8_t6000")
