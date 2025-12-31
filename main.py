from src.autoattacks_utils import compute_autoattacks

model_names = [
    "Peng2023Robust",
    "Rebuffi2021Fixing_70_16_cutmix_ddpm",
    "Wang2020Improving",
    "Rice2020Overfitting",
    "Chen2024Data_WRN_34_20"
]

if __name__ == "__main__":
    results = compute_autoattacks(
        models=model_names,
        samples=150,
        seeds=0,
        batch_size=50,
        mode="fast"
    )
