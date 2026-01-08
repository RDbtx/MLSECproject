import json
import os
import pandas as pd


def results_to_dataframe(results: dict):
    """
    Convert the nested AutoAttack results dictionary into a Pandas DataFrame.
    The returned DataFrame uses model names as the index and epsilon keys as
    columns. Columns are sorted by increasing epsilon values.

    Input:
    - results: nested dictionary produced by the autoattack_models() function.

    Output:
    - df: DataFrame with models as rows and epsilons as columns, containing robust accuracies.

    """
    rows = {}
    for eps_key, model_dict in results.items():
        for model_name, data in model_dict.items():
            rows.setdefault(model_name, {})[eps_key] = data["robust_acc"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df[sorted(df.columns, key=lambda k: int(str(k).split("/")[0]))]
    return df


def save_results_json(results: dict, filename: str, result_dir: str = "./results") -> None:
    """
    Save the raw results dictionary to a JSON file.

    Inputs:
    - results: results dictionary to serialize.
    - filename: base output filename (without extension).
    - result_dir: directory where the JSON file will be saved.

    """
    filename = f"{filename}.json"
    outpath = os.path.join(result_dir, filename)

    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"JSON saved: {os.path.abspath(outpath)}")


def save_results_csv(results: dict, filename: str, result_dir: str = "./results") -> None:
    """
    Save robust accuracies to a CSV file.
    This converts results into a table where each row is a model and each
    column is an epsilon key, then writes it to CSV.

    Inputs:
    - results: nested results dictionary.
    - filename: base output filename (without extension).
    - result_dir: directory where the CSV file will be saved.

    """
    filename = f"{filename}.csv"
    outpath = os.path.join(result_dir, filename)

    df = results_to_dataframe(results)
    df.to_csv(outpath)

    print(f"CSV saved:  {os.path.abspath(outpath)}")


def save_results(results: dict, filename: str) -> None:
    """
    Save AutoAttack results in both JSON and CSV formats.

    Inputs:
    - results: nested results dictionary.
    - filename: base output filename (without extension).

    """
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    save_results_json(results, filename, result_dir=result_dir)
    save_results_csv(results, filename, result_dir=result_dir)
