from pathlib import Path
import runpy


variant = runpy.run_path(str(Path(__file__).with_name("base_r1.py")))["variant"]
CONFIG = variant(
    "r2_selected_tabm_k64_block_dro_sample_seed2027",
    loss_mode="chronological_block_dro",
    tabm_arch_type="tabm",
)
CONFIG["training"]["sample_seed"] = 2027
