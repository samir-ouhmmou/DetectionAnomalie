"""Features par (utilisateur, jour) + étiquettes + découpage temporel."""
import numpy as np
import pandas as pd
from .config import WORK_START, WORK_END, TRAIN_END, VAL_END


def build_user_day(logon: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    d = logon.copy()
    d["day"] = d["date"].dt.normalize()
    d["hour"] = d["date"].dt.hour + d["date"].dt.minute / 60
    d["is_logon"] = (d["activity"] == "Logon").astype(int)
    d["is_logoff"] = 1 - d["is_logon"]
    d["off"] = ((d["hour"] < WORK_START) | (d["hour"] >= WORK_END)).astype(int)
    d["h_logon"] = d["hour"].where(d["is_logon"] == 1)
    d["h_logoff"] = d["hour"].where(d["is_logoff"] == 1)

    # PC "nouveau" de façon CAUSALE : premier jour où l'utilisateur voit ce PC
    first_day = d.groupby(["user", "pc"])["day"].transform("min")
    new_pc = (d.loc[d["day"] == first_day]
                .groupby(["user", "day"])["pc"].nunique().rename("n_new_pc"))

    g = d.groupby(["user", "day"])
    f = g.agg(
        n_logon=("is_logon", "sum"),
        n_logoff=("is_logoff", "sum"),
        n_off_hours=("off", "sum"),
        first_hour=("hour", "min"),
        last_hour=("hour", "max"),
        first_logon_hour=("h_logon", "min"),
        last_logoff_hour=("h_logoff", "max"),
        n_pc=("pc", "nunique"),
    ).reset_index()
    f = f.merge(new_pc.reset_index(), on=["user", "day"], how="left")
    f["n_new_pc"] = f["n_new_pc"].fillna(0)
    f["first_logon_hour"] = f["first_logon_hour"].fillna(f["first_hour"])
    f["last_logoff_hour"] = f["last_logoff_hour"].fillna(f["last_hour"])
    f["span_hours"] = f["last_hour"] - f["first_hour"]
    f["is_weekend"] = (f["day"].dt.dayofweek >= 5).astype(int)
    f["imbalance"] = (f["n_logon"] - f["n_logoff"]).abs()
    f = f.drop(columns=["first_hour", "last_hour"])

    # Étiquettes : (user, jour) contenant au moins un événement malveillant
    bad = labels.assign(day=labels["date"].dt.normalize())[["user", "day"]].drop_duplicates()
    bad["is_attack"] = 1
    f = f.merge(bad, on=["user", "day"], how="left")
    f["is_attack"] = f["is_attack"].fillna(0).astype(int)

    f["split"] = np.where(f["day"] < TRAIN_END, "train",
                  np.where(f["day"] < VAL_END, "val", "test"))
    return add_user_baseline(f)


def add_user_baseline(f: pd.DataFrame) -> pd.DataFrame:
    """Écart de chaque jour par rapport à l'habitude DE L'UTILISATEUR.
    Baseline = médiane et MAD (robustes) calculées sur le TRAIN (jours ouvrés) seulement."""
    tr = f[(f["split"] == "train") & (f["is_weekend"] == 0)]

    def mad(x):
        return 1.4826 * np.median(np.abs(x - np.median(x)))

    g = tr.groupby("user")
    base = pd.DataFrame({
        "med_first": g["first_logon_hour"].median(),
        "mad_first": g["first_logon_hour"].apply(mad),
        "med_last": g["last_logoff_hour"].median(),
        "med_n": g["n_logon"].median(),
    })
    f = f.join(base, on="user")
    f["med_first"] = f["med_first"].fillna(f["first_logon_hour"].median())
    f["med_last"] = f["med_last"].fillna(f["last_logoff_hour"].median())
    f["dev_first_logon"] = (f["first_logon_hour"] - f["med_first"]).abs()
    f["dev_last_logoff"] = (f["last_logoff_hour"] - f["med_last"]).abs()
    f["z_first_logon"] = (f["dev_first_logon"] / f["mad_first"].fillna(1).clip(lower=0.25)).clip(upper=50)
    f["dev_n_logon"] = (f["n_logon"] - f["med_n"].fillna(1)).abs()
    f["row_id"] = np.arange(len(f))
    return f.drop(columns=["med_first", "mad_first", "med_last", "med_n"])
