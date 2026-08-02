from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(
    str(Path(__file__).with_name("r5_tabm_k64_train500k.py"))
)["CONFIG"]
CONFIG = deepcopy(BASE)
CONFIG["training"]["max_train_samples"] = 750_000
CONFIG["output"]["results_name"] = "r6_tabm_k64_train750k"
