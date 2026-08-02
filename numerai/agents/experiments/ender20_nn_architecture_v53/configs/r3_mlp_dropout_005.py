from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(str(Path(__file__).with_name("r1_mlp_big_residual.py")))[
    "CONFIG"
]
CONFIG = deepcopy(BASE)
CONFIG["model"]["params"]["dropout"] = 0.05
CONFIG["output"]["results_name"] = "r3_mlp_dropout_005"
