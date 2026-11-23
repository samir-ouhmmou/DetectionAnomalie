# Détection d'anomalies dans les logs d'authentification (CERT Insider Threat r3.1)

Détection d'un **insider malveillant** (scénario 1 : connexions hors horaires, clé USB, wikileaks)
à partir de `logon.csv` (842 109 événements, 1000 utilisateurs) et de la vérité terrain `r3_1-1.csv`.

> Dataset : CERT Insider Threat (Carnegie Mellon / SEI), version r3.x — **pas** LANL.

## Structure
```
data/r3.1.tar.bz2           archive CERT (le logon.csv est lu directement si non extrait)
data/raw/logon.csv          logs bruts (facultatif si l'archive est présente)
data/labels/r3_1-1.csv      étiquettes (1 insider : CSF0929)
src/config.py               chemins, dates de découpage, hyperparamètres
src/data_check.py           audit qualité des CSV  (python -m src.data_check)
src/ingest.py, features.py  lecture + features par (user, jour) + baseline personnelle
src/models/                 K-Means, chaîne de Markov et Isolation Forest
src/evaluate.py             confusion, P/R/F1, accuracy, PR-AUC, ROC-AUC, P@k, rang (à la main)
src/visualize.py            courbes PR, matrices de confusion
tests_sanity.py             vérifications des métriques et du score K-Means
run_pipeline.py             pipeline complet
results/                    metrics_all_splits.csv, split_summary.csv, data_quality_report.txt, *.png
results/models/              modèles entraînés exportés en .pkl
app.py                       interface Streamlit pour comparer les modèles
```
## Lancer
```bash
pip install -r requirements.txt
python tests_sanity.py
python run_pipeline.py
streamlit run app.py
```
Lancez l'entraînement avant l'interface : elle lit les résultats dans `results/`
et permet de choisir un modèle et le split. Les objets entraînés (avec leur
seuil de décision) sont exportés dans `results/models/`.
Si l'archive n'est pas fournie, placez le fichier `logon.csv` dans `data/raw/`.
Le fichier `r3.1.tar.bz2` présent dans `data/` est également accepté directement ;
seul `logon.csv` est lu et aucune extraction manuelle n'est nécessaire.

Le pipeline entraîne les trois détecteurs, choisit chaque seuil avec le F1 sur
la validation, puis affiche et enregistre accuracy, F1 et TN/FP/FN/TP pour
train/validation/test dans `results/metrics_all_splits.csv`. Les matrices sont
aussi représentées dans `results/confusion_matrices.png`.
## Protocole
- **Unité** : un (utilisateur, jour). Positif = jour contenant un événement du fichier de labels (7 jours).
- **Découpage temporel** : train 2010-01-02 -> 07-03 (130 420 lignes, 3 jours d'attaque) ;
  validation 07-04 -> 07-12 (5 062 lignes, 2 attaques) ; test 07-13 -> 2011-05-16 (204 689 lignes, 2 attaques).
- **Entraînement** : sur le TRAIN seulement. K-Means, Isolation Forest et Markov sont non supervisés.
- **Seuil** : choisi sur la VALIDATION (F1 maximal), puis figé pour le test.
- **Résultats** : matrice de confusion et métriques sur train, validation et test.
- **Anti-fuite** : baseline personnelle (médiane/MAD de l'heure de connexion) et matrices de Markov calculées sur le train ; "PC nouveau" est causal.

## Limites (à citer dans le rapport)
1. Un seul insider, **2 jours positifs dans le test et 2 dans la validation** : F1, précision et PR-AUC sont très instables. La validation (5 062 lignes) est trop petite pour fixer un seuil fiable, ce qui explique les nombreuses fausses alertes en test.
2. L'accuracy dépasse 99,8 % même sans rien détecter : elle est **trompeuse** ici.
3. Le seuil est choisi en maximisant le F1 sur une validation qui contient seulement deux jours d'attaque : ce seuil est instable et ne garantit pas la détection d'autres attaquants.
4. Dataset synthétique ; scénario unique ; `logon.csv` seul ne voit ni les clés USB ni wikileaks (`device.csv`, `http.csv` à ajouter).
5. Résultat expérimental, **pas une garantie de sécurité réelle**.
