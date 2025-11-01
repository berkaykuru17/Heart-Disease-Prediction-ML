# 464_Project.py
# Heart Disease Prediction
# Run ->
#   python 464_Project.py --csv data/heart_disease_uci.csv --out outputs
#   python 464_Project.py --csv data/heart_disease_uci.csv --out outputs --gui

import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
)
import joblib

# ---------------- Core utils ----------------

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "num" not in df.columns:
        raise ValueError("Expected column 'num' not found in CSV.")
    df["target"] = (df["num"] > 0).astype(int)
    return df

def build_preprocess(df: pd.DataFrame, target_col: str = "target"):
    auto_cat = df.select_dtypes(include=["object"]).columns.tolist()
    known_cat = [c for c in ["cp", "restecg", "slope", "thal"] if c in df.columns]
    categorical_cols = sorted(list(set(auto_cat + known_cat)))
    numeric_cols = [c for c in df.columns if c not in categorical_cols + [target_col]]

    num_trans = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_trans = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    prep = ColumnTransformer([
        ("num", num_trans, numeric_cols),
        ("cat", cat_trans, categorical_cols),
    ])
    return prep, numeric_cols, categorical_cols

# ---------------- Training & reporting ----------------

def train_and_eval(df: pd.DataFrame, out_dir: str):
    ensure_dirs(out_dir)

    df = df.drop(columns=[c for c in ["num", "id", "dataset"] if c in df.columns])
    prep, num_cols, cat_cols = build_preprocess(df, "target")
    X, y = df.drop(columns=["target"]), df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    log_pipe = Pipeline([("prep", prep),
                         ("clf", LogisticRegression(max_iter=500, solver="liblinear"))])
    log_grid = {"clf__C": [0.1, 1, 10]}
    log_gs = GridSearchCV(log_pipe, log_grid, cv=cv, scoring="f1", n_jobs=1)
    log_gs.fit(X, y)
    log_best = log_gs.best_estimator_
    log_best.fit(X_train, y_train)
    y_pred_log = log_best.predict(X_test)
    y_prob_log = log_best.predict_proba(X_test)[:, 1]

    rf_pipe = Pipeline([("prep", prep),
                        ("clf", RandomForestClassifier(random_state=42))])
    rf_grid = {"clf__n_estimators": [200], "clf__max_depth": [None, 8], "clf__min_samples_leaf": [1, 2]}
    rf_gs = GridSearchCV(rf_pipe, rf_grid, cv=cv, scoring="f1", n_jobs=1)
    rf_gs.fit(X, y)
    rf_best = rf_gs.best_estimator_
    rf_best.fit(X_train, y_train)
    y_pred_rf = rf_best.predict(X_test)
    y_prob_rf = rf_best.predict_proba(X_test)[:, 1]

    def summarize(y_true, y_pred, y_prob):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

    results = {
        "Logistic Regression": summarize(y_test, y_pred_log, y_prob_log),
        "Random Forest": summarize(y_test, y_pred_rf, y_prob_rf)
    }
    with open(Path(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    RocCurveDisplay.from_estimator(log_best, X_test, y_test)
    plt.title("ROC – Logistic Regression"); plt.tight_layout()
    plt.savefig(Path(out_dir, "roc_logreg.png")); plt.close()

    RocCurveDisplay.from_estimator(rf_best, X_test, y_test)
    plt.title("ROC – Random Forest"); plt.tight_layout()
    plt.savefig(Path(out_dir, "roc_randomforest.png")); plt.close()

    for name, y_pred in [("logreg", y_pred_log), ("rf", y_pred_rf)]:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix – {name.upper()}"); plt.tight_layout()
        plt.savefig(Path(out_dir, f"cm_{name}.png")); plt.close()

    ohe = rf_best.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    cat_features = ohe.get_feature_names_out(build_preprocess(df, "target")[2])
    all_features = list(build_preprocess(df, "target")[1]) + list(cat_features)
    importances = rf_best.named_steps["clf"].feature_importances_
    top = sorted(zip(all_features, importances), key=lambda x: x[1], reverse=True)[:10]
    with open(Path(out_dir, "top_features.json"), "w") as f:
        json.dump(top, f, indent=2)

    joblib.dump(log_best, Path(out_dir, "best_logreg.pkl"))
    joblib.dump(rf_best, Path(out_dir, "best_randomforest.pkl"))

    print("\n==============================")
    print(" HEART DISEASE PREDICTION – SUMMARY")
    print("==============================\n")
    print("Model scores (test set):")
    for m, vals in results.items():
        print(f"\n[{m}]")
        print(f"  Accuracy : {vals['accuracy']*100:.2f}%")
        print(f"  Precision: {vals['precision']*100:.2f}%")
        print(f"  Recall   : {vals['recall']*100:.2f}%")
        print(f"  F1-score : {vals['f1']*100:.2f}%")
        print(f"  ROC-AUC  : {vals['roc_auc']*100:.2f}%")

    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest model: {best}")
    print("\nTop features (Random Forest):")
    for feat, imp in top:
        print(f"  - {feat} ({imp*100:.2f}%)")
    print("\nArtifacts saved to:", Path(out_dir).resolve(), "\n")

    return {
        "best_name": best,
        "best_estimator": rf_best if best == "Random Forest" else log_best,
        "preprocess": prep,
        "X_full": X,
        "df_full": df,
        "results": results
    }

# ---------------- Tkinter GUI ----------------

def launch_gui(df_full: pd.DataFrame, model: Pipeline, out_dir: str):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Heart Disease Risk Dashboard")
    root.geometry("1000x620")

    frame_top = ttk.Frame(root); frame_top.pack(fill="x", padx=10, pady=8)
    frame_mid = ttk.Frame(root); frame_mid.pack(fill="both", expand=True, padx=10, pady=8)
    frame_right = ttk.Frame(root); frame_right.pack(fill="x", padx=10, pady=8)

    lbl = ttk.Label(frame_top, text="Risk scores (0–100%). Select a row for details.")
    lbl.pack(side="left")

    # Compute probabilities for all rows (display purpose)
    X_gui = df_full.drop(columns=["target"])
    probs = model.predict_proba(X_gui)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_gui)
    df_view = df_full.copy().reset_index(drop=True)
    df_view["risk_percent"] = (probs * 100).round(2)

    # Filter controls
    threshold_var = tk.DoubleVar(value=50.0)
    ttk.Label(frame_top, text="Min risk %:").pack(side="left", padx=(12, 4))
    thr_entry = ttk.Entry(frame_top, width=6, textvariable=threshold_var); thr_entry.pack(side="left")
    sort_var = tk.StringVar(value="risk_desc")
    ttk.Checkbutton(frame_top, text="Sort by risk desc", variable=sort_var, onvalue="risk_desc", offvalue="none").pack(side="left", padx=10)

    # Treeview
    cols = ["idx", "age", "sex", "cp", "trestbps", "chol", "thalch", "oldpeak", "target", "risk_percent"]
    present = [c for c in cols if c in df_view.columns] + (["risk_percent"] if "risk_percent" not in cols else [])
    tree = ttk.Treeview(frame_mid, columns=present, show="headings", height=18)
    vsb = ttk.Scrollbar(frame_mid, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    frame_mid.grid_columnconfigure(0, weight=1)
    frame_mid.grid_rowconfigure(0, weight=1)

    for c in present:
        tree.heading(c, text=c)
        tree.column(c, width=90, anchor="center")

    def populate():
        for i in tree.get_children(): tree.delete(i)
        thr = threshold_var.get()
        df_f = df_view[df_view["risk_percent"] >= thr].copy()
        if sort_var.get() == "risk_desc":
            df_f = df_f.sort_values("risk_percent", ascending=False)
        for i, row in df_f.iterrows():
            values = [row.get(c, "") for c in present]
            tree.insert("", "end", values=values)

    populate()

    # Detail panel
    detail = tk.Text(frame_right, height=8)
    detail.pack(fill="x")

    def on_select(_):
        sel = tree.focus()
        if not sel: return
        vals = tree.item(sel, "values")
        rec = dict(zip(present, vals))
        detail.delete("1.0", "end")
        detail.insert("end", "Selected record\n")
        for k, v in rec.items():
            detail.insert("end", f"{k}: {v}\n")
        detail.insert("end", "\nInterpretation:\n")
        rp = float(rec.get("risk_percent", 0))
        if rp >= 70:
            msg = "High risk"
        elif rp >= 40:
            msg = "Moderate risk"
        else:
            msg = "Low risk"
        detail.insert("end", f"Risk band: {msg}\n")

    tree.bind("<<TreeviewSelect>>", on_select)

    # Buttons
    btn_frame = ttk.Frame(frame_top); btn_frame.pack(side="right")
    ttk.Button(btn_frame, text="Apply Filter/Sort", command=populate).pack(side="left", padx=6)
    ttk.Button(btn_frame, text="Export current view (CSV)",
               command=lambda: df_view[df_view["risk_percent"] >= threshold_var.get()]
               .to_csv(Path(out_dir, "gui_current_view.csv"), index=False)).pack(side="left", padx=6)

    root.mainloop()

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/heart_disease_uci.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--gui", action="store_true", help="Launch Tkinter dashboard after training")
    args = ap.parse_args()

    ensure_dirs(args.out, Path(args.out, "eda"))

    df = load_data(args.csv)

    # quick EDA images
    na = df.isna().sum().sort_values(ascending=False)
    plt.figure(); na.plot(kind="bar"); plt.title("Missing Values per Column"); plt.tight_layout()
    plt.savefig(Path(args.out, "eda_missing.png")); plt.close()

    res = train_and_eval(df, args.out)

    if args.gui:
        # Fit the chosen best estimator on the full dataset for GUI inference
        best = res["best_estimator"]
        X_full = res["X_full"]
        y = df["target"]
        best.fit(X_full, y)
        # GUI works with the model pipeline directly on raw X (preprocess inside)
        launch_gui(df, best, args.out)

if __name__ == "__main__":
    main()
