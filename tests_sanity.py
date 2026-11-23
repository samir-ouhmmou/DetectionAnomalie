"""Vérifie que NOS implémentations donnent les mêmes résultats que scikit-learn."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, confusion_matrix
from src.evaluate import average_precision, roc_auc, confusion, prf
from src.models.iforest import IsolationTree, c_factor
from src.models.kmeans import KMeansModel
from src.config import FEATURES

rng = np.random.default_rng(0)
y = (rng.random(5000) < 0.02).astype(int); s = rng.random(5000) + y * 0.6
print("PR-AUC  nôtre/sklearn:", round(average_precision(y, s), 4), round(average_precision_score(y, s), 4))
print("ROC-AUC nôtre/sklearn:", round(roc_auc(y, s), 4), round(roc_auc_score(y, s), 4))
pred = (s > 0.9).astype(int)
tn, fp, fn, tp = confusion(y, pred)
print("confusion identique:", [tn, fp, fn, tp] == list(confusion_matrix(y, pred).ravel()),
      "| F1:", round(prf(tn, fp, fn, tp)[3], 4), round(f1_score(y, pred), 4))
X = rng.normal(size=(3000, 5)); yy = (X[:, 0] + X[:, 1] > 0.5).astype(int)
print("c(256) =", round(float(c_factor(256)), 4), "(attendu ~10.24)")

train = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
test = train.iloc[:10].copy()
test.iloc[0] = 8.0
scores = KMeansModel(n_clusters=3).fit(train).score(test)
assert scores.shape == (len(test),) and np.isfinite(scores).all()
assert scores[0] > np.median(scores[1:])
print("K-Means : distance d'anomalie finie et outlier mieux classé, OK")
