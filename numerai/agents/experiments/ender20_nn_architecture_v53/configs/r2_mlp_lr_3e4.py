from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(str(Path(__file__).with_name("r1_mlp_big_residual.py")))[
    "CONFIG"
]
CONFIG = deepcopy(BASE)
CONFIG["model"]["params"]["learning_rate"] = 3e-4
CONFIG["output"]["results_name"] = "r2_mlp_lr_3e4"
