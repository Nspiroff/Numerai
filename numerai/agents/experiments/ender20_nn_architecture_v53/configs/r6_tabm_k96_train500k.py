from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(
    str(Path(__file__).with_name("r5_tabm_k64_train500k.py"))
)["CONFIG"]
CONFIG = deepcopy(BASE)
CONFIG["model"]["params"]["tabm_k"] = 96
CONFIG["output"]["results_name"] = "r6_tabm_k96_train500k"
