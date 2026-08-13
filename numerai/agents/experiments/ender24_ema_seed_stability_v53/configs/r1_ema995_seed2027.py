from pathlib import Path
import runpy


variant = runpy.run_path(str(Path(__file__).with_name("base_r1.py")))["variant"]
CONFIG = variant("r1_ema995_seed2027", model_seed=2027, ema_decay=0.995)
