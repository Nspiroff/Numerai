from pathlib import Path
import runpy


variant = runpy.run_path(str(Path(__file__).with_name("base_r1.py")))["variant"]
CONFIG = variant(
    "r2_control_tabm_k64_sample_seed2027",
    loss_mode="mse",
    tabm_arch_type="tabm",
)
CONFIG["training"]["sample_seed"] = 2027
