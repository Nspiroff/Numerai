from pathlib import Path
import runpy


variant = runpy.run_path(str(Path(__file__).with_name("base_r1.py")))["variant"]
CONFIG = variant(
    "r2_recent_window78_model_seed2027",
    max_train_eras=78,
    model_seed=2027,
)
