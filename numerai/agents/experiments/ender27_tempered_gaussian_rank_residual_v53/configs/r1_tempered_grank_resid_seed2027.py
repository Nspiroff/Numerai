from pathlib import Path
import runpy


variant = runpy.run_path(str(Path(__file__).with_name("base_r1.py")))["variant"]
CONFIG = variant(
    "r1_tempered_grank_resid_seed2027",
    model_seed=2027,
    benchmark_transform="tie_kept_rank_gaussian",
    benchmark_transform_strength=0.5,
)
