"""Export sklearn pipeline parameters to a Pyodide-compatible pickle.

Loads modelo_turnover.pkl (sklearn 1.8.0) and extracts all parameters
as plain dicts + numpy arrays. Saves as modelo_turnover_pyodide.pkl,
loadable by any scikit-learn version in Pyodide (or elsewhere).

Usage:
    python src/export_pyodide_model.py
"""
import joblib
import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "modelo_turnover.pkl"
DST = ROOT / "modelo_turnover_pyodide.pkl"

pipeline = joblib.load(SRC)

ct = pipeline.named_steps["preprocessor"]
gbc = pipeline.named_steps["classifier"]

num_pipeline = ct.named_transformers_["num"]
cat_pipeline = ct.named_transformers_["cat"]
scaler = num_pipeline.named_steps["scaler"]
encoder = cat_pipeline.named_steps["onehot"]

# -- StandardScaler --
scaler_data = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
}

# -- OneHotEncoder --
encoder_data = {
    "categories": [c.tolist() for c in encoder.categories_],
    "drop_idx": encoder.drop_idx_[0],
}

# -- GradientBoosting trees --
trees = []
for est in gbc.estimators_:
    t = est[0].tree_
    trees.append({
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": t.value.tolist(),  # shape (node_count, 1, 1)
    })

prior = gbc.init_.class_prior_
init_raw = float(np.log(prior[1] / prior[0]))

gbc_data = {
    "init_raw": init_raw,
    "learning_rate": gbc.learning_rate,
    "trees": trees,
}

model_data = {
    "scaler": scaler_data,
    "encoder": encoder_data,
    "gbc": gbc_data,
}

with open(DST, "wb") as f:
    pickle.dump(model_data, f, protocol=4)

print(f"Saved {DST} ({DST.stat().st_size / 1024:.0f} KB)")
