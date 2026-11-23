"""Chaîne de Markov par utilisateur (détection de séquences), écrite à la main.

État d'un événement = (Logon/Logoff, tranche horaire, semaine/week-end) -> 16 états.
Pour chaque utilisateur u on compte les transitions s -> s' sur le TRAIN :
    P_u(s'|s) = (N_u(s,s') + alpha) / (sum_s' N_u(s,s') + alpha * 16)   (lissage de Laplace)
Surprise d'une transition = -log P_u(s'|s). Score d'un (user, jour) = surprise MAX
de ses transitions : une transition rare ou jamais vue est suspecte.
"""
import numpy as np
import pandas as pd
from ..config import TRAIN_END

N_STATES = 16


def encode_states(ev: pd.DataFrame) -> np.ndarray:
    h = ev["date"].dt.hour.values
    bucket = np.select([h < 6, h < 12, h < 18], [0, 1, 2], default=3)   # nuit/matin/après-midi/soir
    act = (ev["activity"] == "Logoff").astype(int).values
    wk = (ev["date"].dt.dayofweek >= 5).astype(int).values
    return act * 8 + bucket * 2 + wk


class MarkovModel:
    name = "Markov (séquences)"

    def __init__(self, logon: pd.DataFrame, alpha: float = 0.5):
        self.ev = logon.sort_values(["user", "date"]).reset_index(drop=True)
        self.alpha = alpha

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("ev", None)
        return state

    def fit(self, train=None):
        ev = self.ev
        ev["state"] = encode_states(ev)
        ev["prev"] = ev.groupby("user")["state"].shift(1)
        ev["day"] = ev["date"].dt.normalize()
        uid = {u: i for i, u in enumerate(ev["user"].unique())}
        ev["uid"] = ev["user"].map(uid)

        tr = ev[(ev["day"] < TRAIN_END) & ev["prev"].notna()]          # comptage sur TRAIN
        N = np.zeros((len(uid), N_STATES, N_STATES))
        np.add.at(N, (tr["uid"].values, tr["prev"].astype(int).values, tr["state"].values), 1)
        self.P = (N + self.alpha) / (N.sum(axis=2, keepdims=True) + self.alpha * N_STATES)

        ok = ev["prev"].notna()
        ev["surprise"] = np.nan
        ev.loc[ok, "surprise"] = -np.log(self.P[ev.loc[ok, "uid"].values,
                                                ev.loc[ok, "prev"].astype(int).values,
                                                ev.loc[ok, "state"].values])
        self.day_scores = (ev.groupby(["user", "day"])["surprise"].max()
                             .rename("markov").reset_index())
        return self

    def score(self, df):
        m = df[["user", "day"]].merge(self.day_scores, on=["user", "day"], how="left")
        return m["markov"].fillna(0).values
