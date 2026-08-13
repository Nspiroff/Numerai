from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(
    str(Path(__file__).with_name("r5_tabm_k64_train500k.py"))
)["CONFIG"]
CONFIG = deepcopy(BASE)

# Scale the scout winner to every consecutive benchmark-covered era while
# preserving the scout's effective 52-original-era outer and inner embargo.
CONFIG["data"].pop("full_data_path", None)
CONFIG["data"].pop("benchmark_data_path", None)
CONFIG["data"].update(
    {
        "disk_feature_store_path": "v5.3/target_ender_20_feature_store",
        "embargo_eras": 52,
    }
)
CONFIG["model"]["params"]["internal_val_embargo"] = 52
CONFIG["training"]["data_mode"] = "disk_feature_store"
CONFIG["training"]["cv"]["embargo"] = 52
CONFIG["output"]["results_name"] = "scale_disk_tabm_k64_train500k"

