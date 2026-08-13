from copy import deepcopy
from pathlib import Path
import runpy


BASE = runpy.run_path(str(Path(__file__).with_name("r1_mlp_big_residual.py")))[
    "CONFIG"
]
CONFIG = deepcopy(BASE)
CONFIG["model"]["params"].update(
    {
        "architecture": "resnet",
        "resnet_width": 512,
        "resnet_hidden_width": 1024,
        "resnet_blocks": 4,
        "dropout": 0.1,
        "batch_size": 2048,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
    }
)
CONFIG["output"]["results_name"] = "r1_resnet_residual"
