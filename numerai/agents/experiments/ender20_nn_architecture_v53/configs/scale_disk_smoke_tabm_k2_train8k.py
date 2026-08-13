from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(
    str(Path(__file__).with_name("r5_tabm_k64_train500k.py"))
)["CONFIG"]
CONFIG = deepcopy(BASE)

# The disk store owns its source manifest and benchmark column.
CONFIG["data"].pop("full_data_path", None)
CONFIG["data"].pop("benchmark_data_path", None)
CONFIG["data"].update(
    {
        "disk_feature_store_path": "v5.3/target_ender_20_feature_store",
        "embargo_eras": 52,
    }
)

# One usable expanding fold over the real consecutive-era store.  This is an
# end-to-end path smoke, not an architecture measurement.
CONFIG["model"]["params"].update(
    {
        "tabm_k": 2,
        "tabm_width": 32,
        "tabm_blocks": 1,
        "max_epochs": 1,
        "patience": 1,
        "internal_val_embargo": 52,
        "prediction_batch_size": 8192,
    }
)
CONFIG["training"].update(
    {
        "data_mode": "disk_feature_store",
        "max_train_samples": 8_192,
    }
)
CONFIG["training"]["cv"].update({"embargo": 52, "n_splits": 2})
CONFIG["output"]["results_name"] = "scale_disk_smoke_tabm_k2_train8k"
