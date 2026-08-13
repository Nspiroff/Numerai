from copy import deepcopy
from pathlib import Path
import runpy


parent = runpy.run_path(
    str(Path(__file__).with_name("r1_tabm_k64_block_dro.py"))
)["CONFIG"]
CONFIG = deepcopy(parent)
CONFIG["output"]["results_name"] = "c1_selected_tabm_k64_block_dro"
