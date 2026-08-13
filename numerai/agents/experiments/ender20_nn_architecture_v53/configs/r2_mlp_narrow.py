from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(str(Path(__file__).with_name("r1_mlp_big_residual.py")))[
    "CONFIG"
]
CONFIG = deepcopy(BASE)
CONFIG["model"]["params"]["hidden_layer_sizes"] = (1024, 768, 512, 256)
CONFIG["output"]["results_name"] = "r2_mlp_narrow"
