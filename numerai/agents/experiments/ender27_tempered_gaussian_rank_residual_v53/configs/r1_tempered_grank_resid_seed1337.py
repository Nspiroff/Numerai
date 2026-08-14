from pathlib import Path
import runpy


variant = runpy.run_path(str(Path(__file__).with_name("base_r1.py")))["variant"]
CONFIG = variant(
    "r1_tempered_grank_resid_seed1337",
    model_seed=1337,
    benchmark_transform="tie_kept_rank_gaussian",
    benchmark_transform_strength=0.5,
)
