from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(str(Path(__file__).with_name("r1_tabm_residual.py")))[
    "CONFIG"
]
CONFIG = deepcopy(BASE)
CONFIG["model"]["params"]["tabm_k"] = 32
CONFIG["output"]["results_name"] = "r3_tabm_k32"
