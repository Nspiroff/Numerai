from pathlib import Path
import runpy


variant = runpy.run_path(str(Path(__file__).with_name("base_r1.py")))["variant"]
CONFIG = variant(
    "r2_recent_half_life52_model_seed2027",
    recency_half_life_eras=52.0,
    model_seed=2027,
)
