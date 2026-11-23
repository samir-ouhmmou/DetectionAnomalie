"""Pipeline complet : python run_pipeline.py"""
import json
import pickle
import numpy as np
import pandas as pd
from src import config as C
from src.ingest import load_logon, load_labels, integrity_report
from src.features import build_user_day
from src.models.iforest import IForestModel
from src.models.kmeans import KMeansModel
from src.models.markov import MarkovModel
from src.evaluate import evaluate, best_f1_threshold
from src.visualize import plot_pr, plot_confusions

C.PROC.mkdir(exist_ok=True); C.RES.mkdir(exist_ok=True); C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

# 1-3. Ingestion, nettoyage, features
logon, labels = load_logon(), load_labels()
rep = integrity_report(logon, labels)
print(json.dumps(rep, indent=2, default=str))
feats = build_user_day(logon, labels)
feats.to_csv(C.PROC / "user_day_features.csv", index=False)
assert feats.duplicated(["user", "day"]).sum() == 0          # pas de doublon -> pas de fuite

# 4. Découpage temporel train / validation / test
S = {s: feats[feats.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
summary = pd.DataFrame([{"split": s, "rows": len(d), "attack_days": int(d.is_attack.sum()),
                         "period": f"{d.day.min():%Y-%m-%d} -> {d.day.max():%Y-%m-%d}"} for s, d in S.items()])
print("\n", summary.to_string(index=False)); summary.to_csv(C.RES / "split_summary.csv", index=False)

# 5. Entraînement (TRAIN) -> seuil choisi sur VALIDATION -> évaluation train/val/test
bad = rep["malicious_users"]
models = [KMeansModel(), MarkovModel(logon), IForestModel()]
artifact_names = {
    KMeansModel: "kmeans.pkl",
    MarkovModel: "markov.pkl",
    IForestModel: "isolation_forest.pkl",
}
results, test_scores = [], {}
for m in models:
    m.fit(S["train"])
    sc = {s: m.score(d) for s, d in S.items()}
    thr, f1_val = best_f1_threshold(S["val"].is_attack.values, sc["val"])
    print(f"[{m.name}] seuil (F1 max sur validation) = {thr:.4f}  (F1 val = {f1_val:.3f})")
    for s, d in S.items():
        results.append(evaluate(d, sc[s], thr, m.name, s, bad))
    test_scores[m.name] = sc["test"]
    artifact = {"name": m.name, "model": m, "threshold": thr, "features": C.FEATURES}
    with (C.MODEL_DIR / artifact_names[type(m)]).open("wb") as output:
        pickle.dump(artifact, output, protocol=pickle.HIGHEST_PROTOCOL)

res = pd.DataFrame(results)
res.to_csv(C.RES / "metrics_all_splits.csv", index=False)
cols = ["model", "split", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1",
        "pr_auc", "roc_auc", "precision@10", "best_rank_day", "insider_user_rank"]
print("\n", res[cols].round(4).to_string(index=False))

plot_pr(S["test"].is_attack.values, test_scores)
plot_confusions(results)
print("\nRésultats dans", C.RES)
