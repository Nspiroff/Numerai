from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(str(Path(__file__).with_name("r3_tabm_k32.py")))["CONFIG"]
CONFIG = deepcopy(BASE)
CONFIG["model"]["params"]["tabm_width"] = 768
CONFIG["output"]["results_name"] = "r4_tabm_width768"
