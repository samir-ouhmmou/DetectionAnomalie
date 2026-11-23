"""Audit qualité des CSV (logon.csv + fichier de labels)."""
import re
import numpy as np
import pandas as pd
from .config import LOGON_FILE, LABEL_FILE, DATE_FMT, RES


def main():
    out = []
    def P(*a):
        s = " ".join(str(x) for x in a); print(s); out.append(s)

    raw = open(LOGON_FILE, "rb").read(2000)
    P("== logon.csv ==")
    P("fin de ligne CRLF (Windows):", b"\r\n" in raw, "-> pandas la gère, OK")
    df = pd.read_csv(LOGON_FILE)
    P("colonnes:", list(df.columns), "| lignes:", len(df))
    P("valeurs manquantes:", int(df.isna().sum().sum()))
    P("id dupliqués:", int(df["id"].duplicated().sum()),
      "| lignes (date,user,pc,activity) dupliquées:",
      int(df.duplicated(["date", "user", "pc", "activity"]).sum()))
    d = pd.to_datetime(df["date"], format=DATE_FMT, errors="coerce")
    P("dates invalides:", int(d.isna().sum()), "| période:", d.min(), "->", d.max())
    P("activités:", df["activity"].value_counts().to_dict())
    P("format user ^[A-Z]{3}\\d{4}$ invalide:", int((~df["user"].str.match(r"^[A-Z]{3}\d{4}$")).sum()),
      "| format pc ^PC-\\d+$ invalide:", int((~df["pc"].str.match(r"^PC-\d+$")).sum()))
    P("utilisateurs:", df["user"].nunique(), "| PC:", df["pc"].nunique())
    ev = df.assign(date=d).sort_values(["user", "date"])
    ev["prev"] = ev.groupby("user")["activity"].shift(1)
    first = ev["prev"].isna()
    P("séquences par user : Logon->Logon:", int(((ev.prev == "Logon") & (ev.activity == "Logon")).sum()),
      "| Logoff->Logoff:", int(((ev.prev == "Logoff") & (ev.activity == "Logoff")).sum()),
      "| 1er événement = Logoff:", int((first & (ev.activity == "Logoff")).sum()),
      "(sessions non appariées = normal dans CERT)")
    P("événements le week-end:", f"{(d.dt.dayofweek >= 5).mean():.1%}")
    nu = df.groupby("pc")["user"].nunique()
    P("PC utilisés par >1 utilisateur:", int((nu > 1).sum()))
    ev_per_user = df.groupby("user").size()
    P("événements/utilisateur min/médiane/max:", ev_per_user.min(), ev_per_user.median(), ev_per_user.max())

    P("\n== r3_1-1.csv (labels) ==")
    lab = pd.read_csv(LABEL_FILE, header=None, names=["type", "id", "date", "user", "pc", "detail"])
    lab["date"] = pd.to_datetime(lab["date"], format=DATE_FMT, errors="coerce")
    P("pas d'en-tête (normal pour ce fichier) | lignes:", len(lab), "| types:", lab["type"].value_counts().to_dict())
    P("dates invalides:", int(lab["date"].isna().sum()), "| utilisateurs:", lab["user"].unique().tolist())
    P("utilisateur présent dans logon.csv:", bool(lab["user"].isin(df["user"]).all()))
    lg = lab[lab["type"] == "logon"].merge(df.assign(date=d), on="id", how="left", suffixes=("", "_l"))
    P("lignes 'logon' des labels retrouvées par id dans logon.csv:", int(lg["user_l"].notna().sum()), "/", len(lg))
    ok = (lg["user"] == lg["user_l"]) & (lg["pc"] == lg["pc_l"]) & (lg["date"] == lg["date_l"])
    P("... et identiques (user, pc, date):", int(ok.sum()), "/", len(lg))
    P("labels 'device' et 'http' : absents de logon.csv (il faut device.csv / http.csv)")
    P("jours d'attaque:", sorted(lab["date"].dt.date.astype(str).unique()))
    P("\nRatio : 1 insider / ", df["user"].nunique(), "utilisateurs")
    (RES / "data_quality_report.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
