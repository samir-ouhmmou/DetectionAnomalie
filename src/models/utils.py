"""Briques de base écrites à la main (numpy uniquement)."""
import numpy as np


class Scaler:
    """Standardisation (x - moyenne) / écart-type, apprise sur le TRAIN."""
    def fit(self, X):
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd == 0, 1.0, sd)
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd
