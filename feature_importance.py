"""
feature_importance.py
Calculates and visualizes clinical feature importance for Breast Cancer survival prediction.
Uses Random Forest and Permutation Feature Importance with cross-validation.
"""

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from preprocess import build_clinical_preprocessor, get_data


def compute_feature_importances(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trains a Gradient Boosting & Random Forest classifier on clinical data,
    evaluating both Gini/Gain feature importance and cross-validated Permutation Importance.
    """
    numeric_features = ["age", "stage"]
    categorical_features = ["chemotherapy", "hormone_therapy", "radiation_therapy", "er_status"]
    feature_cols = numeric_features + categorical_features
    target_col = "vital_status"

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).values

    preprocessor = build_clinical_preprocessor(numeric_features, categorical_features)
    X_trans = preprocessor.fit_transform(X)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names

    # Clean display names
    display_names = {
        "age": "Age at Diagnosis",
        "stage": "Tumor Stage (I-IV / NPI)",
        "chemotherapy_Yes": "Chemotherapy (Yes)",
        "chemotherapy_No": "Chemotherapy (No)",
        "chemotherapy_Unknown": "Chemotherapy (Unknown)",
        "hormone_therapy_Yes": "Hormone Therapy (Yes)",
        "hormone_therapy_No": "Hormone Therapy (No)",
        "hormone_therapy_Unknown": "Hormone Therapy (Unknown)",
        "radiation_therapy_Yes": "Radiation Therapy (Yes)",
        "radiation_therapy_No": "Radiation Therapy (No)",
        "radiation_therapy_Unknown": "Radiation Therapy (Unknown)",
        "er_status_Positive": "ER Receptor (Positive)",
        "er_status_Negative": "ER Receptor (Negative)",
        "er_status_Unknown": "ER Receptor (Unknown)"
    }
    clean_names = [display_names.get(col, col) for col in all_feature_names]

    # Model training
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_trans, y)

    # 1. Native Gini/Gain Importances
    native_importances = model.feature_importances_

    # 2. Permutation Importance across 5-fold CV to avoid overfitting bias
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    perm_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train_cv, y_train_cv = X.iloc[train_idx], y[train_idx]
        X_val_cv, y_val_cv = X.iloc[val_idx], y[val_idx]

        prep_cv = build_clinical_preprocessor(numeric_features, categorical_features)
        X_train_trans = prep_cv.fit_transform(X_train_cv)
        X_val_trans = prep_cv.transform(X_val_cv)

        clf_cv = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
        clf_cv.fit(X_train_trans, y_train_cv)

        r = permutation_importance(clf_cv, X_val_trans, y_val_cv, n_repeats=10, random_state=42, scoring="roc_auc")
        perm_scores.append(r.importances_mean)

    perm_mean = np.mean(perm_scores, axis=0)
    perm_std = np.std(perm_scores, axis=0)

    # Detailed Feature-Level DataFrame
    df_importance = pd.DataFrame({
        "feature_raw": all_feature_names,
        "feature_name": clean_names,
        "native_importance": native_importances,
        "permutation_mean": perm_mean,
        "permutation_std": perm_std
    }).sort_values(by="permutation_mean", ascending=False).reset_index(drop=True)

    # 3. Aggregate importance by original high-level clinical variable
    group_map = {}
    for raw in all_feature_names:
        for orig in feature_cols:
            if raw == orig or raw.startswith(orig + "_"):
                group_map[raw] = orig
                break

    df_importance["clinical_group"] = df_importance["feature_raw"].map(group_map)
    group_names = {
        "age": "Age at Diagnosis",
        "stage": "Tumor Stage",
        "hormone_therapy": "Hormone Therapy",
        "chemotherapy": "Chemotherapy",
        "er_status": "ER Receptor Status",
        "radiation_therapy": "Radiation Therapy"
    }

    df_grouped = df_importance.groupby("clinical_group").agg({
        "permutation_mean": "sum",
        "native_importance": "sum"
    }).reset_index()
    df_grouped["clinical_feature"] = df_grouped["clinical_group"].map(group_names)
    df_grouped = df_grouped.sort_values(by="permutation_mean", ascending=False).reset_index(drop=True)

    return df_importance, df_grouped


def plot_feature_importance(df_grouped: pd.DataFrame, df_detailed: pd.DataFrame, output_path: str = "feature_importance.png"):
    """
    Plots a high-resolution, publication-ready two-panel feature importance bar chart.
    Left: High-level Clinical Feature Groups.
    Right: Detailed Transformed/Encoded Clinical Indicators.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # --- Panel 1: Aggregated Clinical Variables ---
    sns.barplot(
        data=df_grouped,
        y="clinical_feature",
        x="permutation_mean",
        hue="clinical_feature",
        palette="viridis",
        legend=False,
        ax=ax1,
        edgecolor="black",
        linewidth=0.8
    )

    for i, p in enumerate(ax1.patches):
        width = p.get_width()
        val_pct = (width / df_grouped["permutation_mean"].sum()) * 100
        ax1.text(
            width + 0.001,
            p.get_y() + p.get_height() / 2,
            f"{width:.4f} ({val_pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="semibold"
        )

    ax1.set_title("A. High-Level Clinical Attributes (Permutation Impact)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Mean Drop in ROC-AUC upon Permutation", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Clinical Attribute", fontsize=11, fontweight="semibold")
    ax1.set_xlim(0, df_grouped["permutation_mean"].max() * 1.35)

    # --- Panel 2: Detailed Sub-level Indicators ---
    top_detailed = df_detailed.head(8).sort_values(by="permutation_mean", ascending=True)
    y_pos = np.arange(len(top_detailed))

    ax2.barh(
        y_pos,
        top_detailed["permutation_mean"],
        xerr=top_detailed["permutation_std"],
        color="#2b5c8f",
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        alpha=0.85
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_detailed["feature_name"], fontsize=10)

    for i, (mean_val, std_val) in enumerate(zip(top_detailed["permutation_mean"], top_detailed["permutation_std"])):
        ax2.text(
            mean_val + std_val + 0.001,
            i,
            f"{mean_val:.4f}",
            va="center",
            ha="left",
            fontsize=9.5
        )

    ax2.set_title("B. Top Sub-level Features (Mean ± Std CV Permutation AUC)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Mean Drop in ROC-AUC", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("")
    ax2.set_xlim(0, (top_detailed["permutation_mean"] + top_detailed["permutation_std"]).max() * 1.35)

    plt.suptitle("Clinical Feature Importance in Cancer Meta-Analysis Pipeline", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved to: {output_path}")


def main():
    print("--- 1. Ingesting Harmonized BRCA Clinical Dataset ---")
    data = get_data("brca_clinical_data.csv")
    print(f"Loaded {len(data)} patient records.")

    print("\n--- 2. Computing Feature Importances (Model-based & 5-Fold Permutation) ---")
    df_detailed, df_grouped = compute_feature_importances(data)

    print("\n========================================================")
    print("       TOP 3 MOST IMPORTANT CLINICAL FEATURES           ")
    print("========================================================")
    for rank, row in df_grouped.head(3).iterrows():
        pct = (row["permutation_mean"] / df_grouped["permutation_mean"].sum()) * 100
        print(f"  #{rank+1}. {row['clinical_feature']:<25s} | Impact Score: {row['permutation_mean']:.4f} ({pct:.1f}% relative impact)")
    print("========================================================\n")

    print("Complete Feature Ranking Table:")
    print(df_grouped.to_string(index=False))

    print("\n--- 3. Generating High-Resolution Feature Importance Plot ---")
    plot_feature_importance(df_grouped, df_detailed, output_path="feature_importance.png")


if __name__ == "__main__":
    main()
