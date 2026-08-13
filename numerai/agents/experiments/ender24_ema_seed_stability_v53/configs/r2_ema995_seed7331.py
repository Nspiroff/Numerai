from pathlib import Path
import runpy


variant = runpy.run_path(str(Path(__file__).with_name("base_r1.py")))["variant"]
CONFIG = variant("r2_ema995_seed7331", model_seed=7331, ema_decay=0.995)
