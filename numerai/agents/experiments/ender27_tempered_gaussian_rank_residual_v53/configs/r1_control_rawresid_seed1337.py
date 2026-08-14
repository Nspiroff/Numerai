from pathlib import Path
import runpy


variant = runpy.run_path(str(Path(__file__).with_name("base_r1.py")))["variant"]
CONFIG = variant("r1_control_rawresid_seed1337", model_seed=1337)
