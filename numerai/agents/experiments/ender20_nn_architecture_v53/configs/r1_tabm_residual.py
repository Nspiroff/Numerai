from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(str(Path(__file__).with_name("r1_mlp_big_residual.py")))[
    "CONFIG"
]
CONFIG = deepcopy(BASE)
CONFIG["model"]["params"].update(
    {
        "architecture": "tabm",
        "activation": "relu",
        "tabm_arch_type": "tabm",
        "tabm_k": 16,
        "tabm_width": 512,
        "tabm_blocks": 3,
        "dropout": 0.1,
        "batch_size": 1024,
        "prediction_batch_size": 2048,
        "learning_rate": 2e-3,
        "weight_decay": 3e-4,
    }
)
CONFIG["output"]["results_name"] = "r1_tabm_residual"
