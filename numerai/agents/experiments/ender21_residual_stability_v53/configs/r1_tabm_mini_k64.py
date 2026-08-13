from pathlib import Path
import runpy


variant = runpy.run_path(str(Path(__file__).with_name("base_r1.py")))["variant"]
CONFIG = variant("r1_tabm_mini_k64", loss_mode="mse", tabm_arch_type="tabm-mini")
