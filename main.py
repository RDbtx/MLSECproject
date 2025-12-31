import json
from src.autoattacks_utils import compute_autoattacks
from src.results_analysis_utils import make_all_plots

robust_bench_models = {
    "Peng2023Robust": 0.7371,
    "Rebuffi2021Fixing_70_16_cutmix_ddpm": 0.6658,
    "Wang2020Improving": 0.6041,
    "Rice2020Overfitting": 0.5342,
    "Chen2024Data_WRN_34_20": 0.5809
}

if __name__ == "__main__":

    model_names = robust_bench_models.keys()

    """results = compute_autoattacks(
        models=model_names
        samples=150,
        seeds=0,
        batch_size=50,
        mode="fast"
        out_file_name="results"
    )"""

    make_all_plots("results.csv", rb_acc=robust_bench_models)
