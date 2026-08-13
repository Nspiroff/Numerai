from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(
    str(Path(__file__).with_name("r1_base_d6_t6000.py"))
)["CONFIG"]
CONFIG = deepcopy(BASE)
CONFIG["model"]["params"].update({"max_depth": 8, "num_leaves": 255})
CONFIG["output"]["results_name"] = "r1_depth8"
