"""K-Means : score d'anomalie basé sur la distance au centroïde le plus proche."""
import numpy as np
from sklearn.cluster import KMeans
from ..config import FEATURES, SEED
from .utils import Scaler


class KMeansModel:
    name = "K-Means"

    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters

    def fit(self, train):
        X = train[FEATURES].values.astype(float)
        self.sc = Scaler().fit(X)
        X = self.sc.transform(X)
        self.model = KMeans(
            n_clusters=min(self.n_clusters, len(X)),
            random_state=SEED,
            n_init=10,
        ).fit(X)
        return self

    def score(self, df):
        X = self.sc.transform(df[FEATURES].values.astype(float))
        return self.model.transform(X).min(axis=1)
