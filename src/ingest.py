"""Lecture, nettoyage et contrôles d'intégrité."""
import bz2
import io
import tarfile
import pandas as pd
from .config import DATA_ARCHIVE, LOGON_FILE, LABEL_FILE, DATE_FMT


def load_logon(path=LOGON_FILE) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = None
        if DATA_ARCHIVE.exists():
            with bz2.BZ2File(DATA_ARCHIVE, "rb") as compressed:
                with tarfile.open(fileobj=compressed, mode="r|") as archive:
                    for member in archive:
                        if member.isfile() and member.name.endswith("/logon.csv"):
                            source = archive.extractfile(member)
                            if source is None:
                                raise OSError(f"Impossible de lire {member.name} dans {DATA_ARCHIVE}")
                            df = pd.read_csv(io.BytesIO(source.read(member.size)))
                            break
        if df is None:
            raise FileNotFoundError(
                f"Fichier de logs introuvable : {path}. "
                f"Placez logon.csv dans data/raw/ ou l'archive {DATA_ARCHIVE}."
            )
    df["date"] = pd.to_datetime(df["date"], format=DATE_FMT)
    n0 = len(df)
    df = df.drop_duplicates(subset="id")          # doublons d'événements
    df = df.dropna()
    df = df.sort_values(["user", "date"]).reset_index(drop=True)
    df.attrs["removed_rows"] = n0 - len(df)
    return df


def load_labels(path=LABEL_FILE) -> pd.DataFrame:
    cols = ["type", "id", "date", "user", "pc", "detail"]
    lab = pd.read_csv(path, header=None, names=cols)
    lab["date"] = pd.to_datetime(lab["date"], format=DATE_FMT)
    return lab


def integrity_report(logon: pd.DataFrame, labels: pd.DataFrame) -> dict:
    return {
        "n_events": len(logon),
        "n_users": logon["user"].nunique(),
        "n_pcs": logon["pc"].nunique(),
        "date_min": str(logon["date"].min()),
        "date_max": str(logon["date"].max()),
        "removed_duplicates_or_na": logon.attrs.get("removed_rows", 0),
        "malicious_users": sorted(labels["user"].unique()),
        "n_label_events": len(labels),
    }
