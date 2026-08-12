from pathlib import Path
import runpy


variant = runpy.run_path(str(Path(__file__).with_name("base_r1.py")))["variant"]
CONFIG = variant(
    "r1_tabm_mini_k64_block_dro",
    loss_mode="chronological_block_dro",
    tabm_arch_type="tabm-mini",
)
