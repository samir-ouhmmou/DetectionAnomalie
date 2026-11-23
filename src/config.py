"""Configuration centrale du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LABELS = ROOT / "data" / "labels"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
MODEL_DIR = RES / "models"
DATA_ARCHIVE = ROOT / "data" / "r3.1.tar.bz2"

DATE_FMT = "%m/%d/%Y %H:%M:%S"
LOGON_FILE = RAW / "logon.csv"
LABEL_FILE = LABELS / "r3_1-1.csv"   # type,id,date,user,pc,detail (sans en-tête)

# Découpage TEMPOREL (jamais aléatoire : on évite de mélanger futur et passé)
# Les 7 jours d'attaque sont répartis : train 3 (1-3 juil.), val 2 (8-9), test 2 (14, 16)
TRAIN_END = "2010-07-04"   # train : jour < 2010-07-04   (jan -> 3 juil.)
VAL_END = "2010-07-13"     # val   : 2010-07-04 <= jour < 2010-07-13
                           # test  : jour >= 2010-07-13 (-> mai 2011)

# Heures de travail (en dehors = "hors horaires")
WORK_START, WORK_END = 7, 19

# Seuil des modèles non supervisés : quantile des scores de VALIDATION
THRESHOLD_QUANTILE = 0.999

SEED = 42

FEATURES = [
    "n_logon", "n_logoff", "n_off_hours", "first_logon_hour",
    "last_logoff_hour", "span_hours", "n_pc", "n_new_pc",
    "is_weekend", "imbalance",
    # écarts par rapport à la baseline personnelle de l'utilisateur
    "dev_first_logon", "dev_last_logoff", "z_first_logon", "dev_n_logon",
]
