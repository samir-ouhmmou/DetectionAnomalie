"""Isolation Forest implémenté à la main (Liu, Ting & Zhou, 2008).

Idée : une anomalie est rare et différente -> elle est isolée avec peu de
coupures aléatoires. Chaque arbre coupe au hasard (variable aléatoire, seuil
aléatoire entre min et max) jusqu'à isoler les points.
Score = 2^(-E[h(x)] / c(psi)) ; proche de 1 = anormal, < 0,5 = normal.
"""
import numpy as np
from ..config import FEATURES, SEED
from .utils import Scaler

EULER = 0.5772156649


def c_factor(n):
    """Longueur moyenne d'un chemin dans un arbre binaire de recherche à n points."""
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + EULER) - 2.0 * (n[big] - 1.0) / n[big]
    out[n == 2] = 1.0
    return out


class IsolationTree:
    def __init__(self, max_depth, rng):
        self.max_depth, self.rng = max_depth, rng

    def fit(self, X):
        feat, thr, left, right, size = [], [], [], [], []
        stack = [(np.arange(len(X)), 0, -1, None)]   # (indices, profondeur, parent, côté)
        while stack:
            idx, depth, parent, side = stack.pop()
            nid = len(feat)
            if parent >= 0:
                (left if side == "L" else right)[parent] = nid
            feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1)
            size.append(len(idx))
            if depth >= self.max_depth or len(idx) <= 1:
                continue
            sub = X[idx]
            lo, hi = sub.min(axis=0), sub.max(axis=0)
            cand = np.flatnonzero(hi > lo)            # variables non constantes
            if len(cand) == 0:
                continue
            q = self.rng.choice(cand)                 # variable choisie au hasard
            p = self.rng.uniform(lo[q], hi[q])        # seuil choisi au hasard
            feat[nid], thr[nid] = q, p
            mask = sub[:, q] < p
            stack.append((idx[mask], depth + 1, nid, "L"))
            stack.append((idx[~mask], depth + 1, nid, "R"))
        self.feat, self.thr = np.array(feat), np.array(thr)
        self.left, self.right, self.size = np.array(left), np.array(right), np.array(size)
        return self

    def path_length(self, X):
        n = len(X)
        node = np.zeros(n, dtype=int)
        depth = np.zeros(n)
        for _ in range(self.max_depth + 1):
            f = self.feat[node]
            active = f >= 0
            if not active.any():
                break
            go_left = X[np.arange(n), np.where(active, f, 0)] < self.thr[node]
            nxt = np.where(go_left, self.left[node], self.right[node])
            node = np.where(active, nxt, node)
            depth += active
        return depth + c_factor(self.size[node])      # correction au niveau des feuilles


class IForestModel:
    name = "Isolation Forest"

    def __init__(self, n_trees=200, psi=256):
        self.n_trees, self.psi = n_trees, psi

    def fit(self, train):
        rng = np.random.default_rng(SEED)
        X = train[FEATURES].values.astype(float)
        self.sc = Scaler().fit(X)
        X = self.sc.transform(X)
        psi = min(self.psi, len(X))
        self.c = float(c_factor(psi))
        max_depth = int(np.ceil(np.log2(psi)))
        self.trees = []
        for _ in range(self.n_trees):
            sample = X[rng.choice(len(X), psi, replace=False)]
            self.trees.append(IsolationTree(max_depth, rng).fit(sample))
        return self

    def score(self, df):
        X = self.sc.transform(df[FEATURES].values.astype(float))
        mean_h = np.mean([t.path_length(X) for t in self.trees], axis=0)
        return 2.0 ** (-mean_h / self.c)
