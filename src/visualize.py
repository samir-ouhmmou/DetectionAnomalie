"""Graphiques : courbes PR (test) et matrices de confusion (train / val / test)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .config import RES


def plot_pr(y, scores: dict, path=RES / "pr_curves_test.png"):
    plt.figure(figsize=(6, 4.5))
    for name, s in scores.items():
        o = np.argsort(-s, kind="stable"); yy = y[o]
        tp = np.cumsum(yy)
        plt.plot(tp / max(1, yy.sum()), tp / np.arange(1, len(yy) + 1), label=name)
    plt.xlabel("Rappel"); plt.ylabel("Précision"); plt.title("Courbes Précision-Rappel (TEST)")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_confusions(results: list, path=RES / "confusion_matrices.png"):
    models = list(dict.fromkeys(r["model"] for r in results))
    splits = ["train", "val", "test"]
    fig, axes = plt.subplots(len(models), 3, figsize=(9, 2.9 * len(models)))
    for i, m in enumerate(models):
        for j, s in enumerate(splits):
            r = next(x for x in results if x["model"] == m and x["split"] == s)
            cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
            ax = axes[i, j]
            ax.imshow(np.log1p(cm), cmap="Blues")
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{cm[a, b]:,}", ha="center", va="center", fontsize=9)
            ax.set_xticks([0, 1], ["Normal", "Alerte"]); ax.set_yticks([0, 1], ["Normal", "Attaque"])
            ax.set_title(f"{m} - {s}", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
