import json
import os
import pandas as pd


def results_to_dataframe(results: dict) -> pd.DataFrame:
    rows = {}
    for eps_key, model_dict in results.items():
        for model_name, val in model_dict.items():
            acc = val["robust_acc"] if isinstance(val, dict) else float(val)
            rows.setdefault(model_name, {})[eps_key] = acc

    df = pd.DataFrame.from_dict(rows, orient="index")

    def eps_num(k):
        try:
            return int(str(k).split("/")[0])
        except Exception:
            return 10 ** 9

    df = df[sorted(df.columns, key=eps_num)]
    return df

def save_results_json(results: dict, result_dir: str = "./results") -> None:
    filename = "results.json"
    outpath = os.path.join(result_dir, filename)

    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"JSON saved: {os.path.abspath(outpath)}")


def save_results_csv(results: dict, result_dir: str = "./results") -> None:
    filename = "results.csv"
    outpath = os.path.join(result_dir, filename)

    df = results_to_dataframe(results)
    df.to_csv(outpath)

    print(f"CSV saved:  {os.path.abspath(outpath)}")


def save_results(results: dict) -> None:
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    save_results_json(results, result_dir=result_dir)
    save_results_csv(results, result_dir=result_dir)
