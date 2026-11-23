"""Métriques écrites à la main (numpy/pandas) : confusion, P/R/F1, accuracy, PR-AUC, ROC-AUC..."""
import numpy as np
import pandas as pd


def confusion(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    return tn, fp, fn, tp


def prf(tn, fp, fn, tp):
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return acc, p, r, f1


def average_precision(y, s):
    """PR-AUC = moyenne des précisions aux rangs des positifs."""
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="stable")
    yy = y[o]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float(prec[yy == 1].mean())


def roc_auc(y, s):
    """AUC via la statistique de Mann-Whitney (rangs moyens pour les égalités)."""
    n1, n0 = int(y.sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_f1_threshold(y, s):
    """Seuil maximisant le F1 sur la VALIDATION (parcours de tous les seuils possibles)."""
    o = np.argsort(-s, kind="stable")
    ss, yy = s[o], y[o]
    tp = np.cumsum(yy); k = np.arange(1, len(yy) + 1)
    p, r = tp / k, tp / max(1, yy.sum())
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
    last_of_tie = np.r_[ss[1:] != ss[:-1], True]       # on ne coupe pas au milieu d'égalités
    f1 = np.where(last_of_tie, f1, -1)
    i = int(np.argmax(f1))
    return float(ss[i]), float(f1[i])


def evaluate(df, score, threshold, name, split, malicious_users=(), ks=(10, 50, 100)):
    y = df["is_attack"].values.astype(int)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion(y, pred)
    acc, p, r, f1 = prf(tn, fp, fn, tp)
    order = np.argsort(-score, kind="stable")
    ranks = np.empty(len(score), dtype=int); ranks[order] = np.arange(1, len(score) + 1)
    res = {"model": name, "split": split, "n_rows": len(y), "n_pos": int(y.sum()),
           "threshold": threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
           "accuracy": acc, "precision": p, "recall": r, "f1": f1,
           "pr_auc": average_precision(y, score), "roc_auc": roc_auc(y, score),
           "best_rank_day": int(ranks[y == 1].min()) if y.sum() else None}
    for k in ks:
        res[f"precision@{k}"] = float(y[order[:k]].mean())
    u = pd.DataFrame({"user": df["user"].values, "s": score}).groupby("user")["s"].max()
    ur = u.rank(ascending=False, method="min")
    for mu in malicious_users:
        if mu in ur.index:
            res["insider_user_rank"] = int(ur[mu]); res["n_users"] = int(len(ur))
    return res
