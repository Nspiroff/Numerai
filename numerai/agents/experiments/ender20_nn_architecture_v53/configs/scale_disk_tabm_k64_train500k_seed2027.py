from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(
    str(Path(__file__).with_name("scale_disk_tabm_k64_train500k.py"))
)["CONFIG"]
CONFIG = deepcopy(BASE)
CONFIG["model"]["params"]["seed"] = 2027
CONFIG["output"]["results_name"] = "scale_disk_tabm_k64_train500k_seed2027"
