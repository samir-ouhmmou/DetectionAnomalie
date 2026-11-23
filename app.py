"""Interface Streamlit pour explorer les résultats des modèles."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS_FILE = ROOT / "results" / "metrics_all_splits.csv"
MODEL_FILES = {
    "K-Means": "kmeans.pkl",
    "Markov (séquences)": "markov.pkl",
    "Isolation Forest": "isolation_forest.pkl",
}

st.set_page_config(page_title="Détection d'anomalies", page_icon="📊", layout="wide")
st.title("Détection d'activités inhabituelles")
st.caption("Comparaison des modèles K-Means, Markov et Isolation Forest.")

if not METRICS_FILE.exists():
    st.error("Résultats absents. Lancez d'abord `python run_pipeline.py`.")
    st.stop()

metrics = pd.read_csv(METRICS_FILE)
splits = [split for split in ("test", "val", "train") if split in metrics["split"].values]
selected_split = st.selectbox("Jeu évalué", splits, format_func=lambda value: {
    "test": "Test", "val": "Validation", "train": "Entraînement"
}[value])
split_metrics = metrics[metrics["split"] == selected_split].copy()

st.subheader("Comparaison des modèles")
comparison = split_metrics[["model", "accuracy", "f1", "TN", "FP", "FN", "TP"]].copy()
comparison["accuracy"] = comparison["accuracy"].map(lambda value: f"{value:.2%}")
comparison["f1"] = comparison["f1"].map(lambda value: f"{value:.2%}")
st.dataframe(
    comparison.rename(columns={
        "model": "Modèle", "accuracy": "Accuracy", "f1": "F1",
        "TN": "Vrais négatifs", "FP": "Fausses alertes",
        "FN": "Attaques manquées", "TP": "Attaques détectées",
    }),
    hide_index=True,
    width="stretch",
)

selected_model = st.selectbox("Modèle à visualiser", split_metrics["model"].tolist())
result = split_metrics[split_metrics["model"] == selected_model].iloc[0]
artifact_path = ROOT / "results" / "models" / MODEL_FILES[selected_model]

accuracy_col, f1_col, precision_col, recall_col = st.columns(4)
accuracy_col.metric("Accuracy", f"{result['accuracy']:.2%}")
f1_col.metric("F1 score", f"{result['f1']:.2%}")
precision_col.metric("Précision", f"{result['precision']:.2%}")
recall_col.metric("Rappel", f"{result['recall']:.2%}")
if artifact_path.exists():
    st.caption(f"Modèle entraîné : `{artifact_path.relative_to(ROOT)}`")
else:
    st.warning("Fichier .pkl absent. Relancez `python run_pipeline.py` pour l'exporter.")

matrix = np.array([
    [int(result["TN"]), int(result["FP"])],
    [int(result["FN"]), int(result["TP"])],
])
figure, axis = plt.subplots(figsize=(5, 4))
image = axis.imshow(matrix, cmap="Blues")
axis.set(
    xticks=[0, 1], yticks=[0, 1],
    xticklabels=["Normal", "Anomalie"],
    yticklabels=["Normal", "Anomalie"],
    xlabel="Prédiction", ylabel="Vérité terrain",
    title=f"Matrice de confusion — {selected_model}",
)
threshold = matrix.max() / 2 if matrix.size else 0
for row in range(2):
    for column in range(2):
        axis.text(
            column, row, f"{matrix[row, column]:,}",
            ha="center", va="center",
            color="white" if matrix[row, column] > threshold else "black",
        )
figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
figure.tight_layout()
st.pyplot(figure, clear_figure=True)
plt.close(figure)
