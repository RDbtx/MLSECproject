import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def eps_value(eps_key):
    s = str(eps_key).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def load_acc_csv(csv_path="results.csv"):
    df = pd.read_csv(csv_path, index_col=0)  # rows=models, cols=eps keys
    df = df.loc[:, sorted(df.columns, key=eps_value)]
    return df


def plot_acc_vs_eps(df_acc, savepath, title="Robust accuracy vs ε"):
    x = np.array([eps_value(e) for e in df_acc.columns], dtype=float)

    plt.figure()
    for m in df_acc.index:
        plt.plot(x, df_acc.loc[m].values.astype(float), marker="o", label=m)

    plt.xlabel("ε")
    plt.ylabel("Robust accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_rank_vs_eps(df_acc, savepath, title="Rank position vs ε"):
    df_rank = df_acc.rank(axis=0, ascending=False, method="average")
    x = np.array([eps_value(e) for e in df_rank.columns], dtype=float)

    plt.figure()
    for m in df_rank.index:
        plt.plot(x, df_rank.loc[m].values.astype(float), marker="o", label=m)

    plt.xlabel("ε")
    plt.ylabel("Rank (1 = best)")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_heatmap_acc(df_acc, savepath, title="Heatmap: robust accuracy (models × ε)"):
    df_rank = df_acc.rank(axis=0, ascending=False, method="average")
    order = df_rank.mean(axis=1).sort_values().index
    df_plot = df_acc.loc[order]

    plt.figure()
    plt.imshow(df_plot.values.astype(float), aspect="auto")
    plt.colorbar(label="Robust accuracy")
    plt.yticks(range(df_plot.shape[0]), df_plot.index, fontsize=8)
    plt.xticks(range(df_plot.shape[1]), df_plot.columns, rotation=45, ha="right")
    plt.xlabel("ε")
    plt.ylabel("Model")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_acc_vs_eps_with_rb_points(df_acc, rb_acc, savepath, rb_eps="8/255",
                                   title="Robust accuracy vs ε (ours) + RobustBench point"):
    x = np.array([eps_value(e) for e in df_acc.columns], dtype=float)
    x_rb = eps_value(rb_eps)

    plt.figure()
    for m in df_acc.index:
        plt.plot(x, df_acc.loc[m].values.astype(float), marker="o", label=m)
        if m in rb_acc and rb_acc[m] is not None:
            plt.scatter([x_rb], [float(rb_acc[m])], marker="x")

    plt.xlabel("ε")
    plt.ylabel("Robust accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_rb_vs_ours_at_eps(df_acc, rb_acc, savepath, rb_eps="8/255",
                           title="RobustBench vs our AutoAttack robust accuracy"):
    col = rb_eps if rb_eps in df_acc.columns else min(df_acc.columns,
                                                      key=lambda c: abs(eps_value(c) - eps_value(rb_eps)))

    models = list(rb_acc.keys())
    rb_vals = [float(rb_acc[m]) for m in models]
    our_vals = [float(df_acc.loc[m, col]) for m in models]

    x = np.arange(len(models))
    w = 0.38

    plt.figure()
    plt.bar(x - w / 2, rb_vals, w, label="RobustBench")
    plt.bar(x + w / 2, our_vals, w, label="Ours")

    plt.xticks(x, models, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Robust accuracy")
    plt.title(f"{title}\n(ours at ε={col}, RB at ε={rb_eps})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def make_all_plots(result_csv_filename="results.csv", rb_acc=None, rb_eps="8/255") -> None:
    out_dir = "./plots"
    results_dir = "./results"

    name = result_csv_filename.replace(".csv", "")
    os.makedirs(out_dir, exist_ok=True)

    df_acc = load_acc_csv(os.path.join(results_dir, result_csv_filename))

    if rb_acc:
        plot_acc_vs_eps_with_rb_points(
            df_acc, rb_acc, os.path.join(out_dir, f"{name}_acc_vs_eps_with_rb.png"),
            rb_eps=rb_eps
        )
        plot_rb_vs_ours_at_eps(
            df_acc, rb_acc, os.path.join(out_dir, f"{name}rb_vs_ours.png"),
            rb_eps=rb_eps
        )
    else:
        plot_acc_vs_eps(df_acc, os.path.join(out_dir, f"{name}_acc_vs_eps.png"),
                        title="Robust accuracy vs AutoAttack ε ")

    plot_rank_vs_eps(df_acc, os.path.join(out_dir, f"{name}_rank_vs_eps.png"))
    plot_heatmap_acc(df_acc, os.path.join(out_dir, f"{name}_heatmap_acc.png"))
