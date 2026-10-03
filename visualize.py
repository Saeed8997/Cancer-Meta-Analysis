"""
visualize.py
Generates publication-ready visualizations for Cancer Meta-Analysis:
1. Kaplan-Meier Survival Curves comparing TCGA and METABRIC cohorts with Log-Rank test.
2. Forest Plot illustrating cohort-specific AUCs and the Pooled Meta-Analysis estimate with 95% CIs.
3. Corrected Forest Plot (forest_plot_corrected.png) isolating robust biological features (Age, Stage, ER Status).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from preprocess import get_data
from model import CancerMetaAnalysisPipeline


def plot_kaplan_meier(df: pd.DataFrame, output_path: str = "kaplan_meier.png"):
    """
    Plots high-resolution Kaplan-Meier survival curves comparing cohorts with log-rank testing.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    kmf = KaplanMeierFitter()
    colors = {"brca_metabric": "#1f77b4", "brca_tcga_pan_can_atlas_2018": "#d62728"}
    labels = {
        "brca_metabric": "METABRIC Cohort",
        "brca_tcga_pan_can_atlas_2018": "TCGA PanCancer Atlas"
    }

    cohort_groups = []
    durations_list = []
    events_list = []

    for study_id in df["study_id"].unique():
        sub_df = df[df["study_id"] == study_id]
        durations = sub_df["overall_survival_months"]
        events = sub_df["vital_status"]

        cohort_groups.append(study_id)
        durations_list.append(durations)
        events_list.append(events)

        kmf.fit(durations, event_observed=events, label=f"{labels.get(study_id, study_id)} (N={len(sub_df)})")
        kmf.plot_survival_function(
            ax=ax,
            color=colors.get(study_id, "#333333"),
            linewidth=2.5,
            ci_alpha=0.15,
            show_censors=True,
            censor_styles={"marker": "|", "ms": 5, "mew": 1.2}
        )

    # Perform Log-rank test between the two cohorts
    if len(cohort_groups) == 2:
        lr_result = logrank_test(
            durations_list[0], durations_list[1],
            event_observed_A=events_list[0], event_observed_B=events_list[1]
        )
        p_val_str = f"p < 0.001" if lr_result.p_value < 0.001 else f"p = {lr_result.p_value:.4f}"
        ax.text(
            0.04, 0.15,
            f"Log-rank Test: $\\chi^2$ = {lr_result.test_statistic:.2f}\n{p_val_str}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#cccccc")
        )

    ax.set_title("Kaplan-Meier Overall Survival by Cohort (BRCA Meta-Analysis)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Time (Months)", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Overall Survival Probability", fontsize=12, fontweight="semibold")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0, max(df["overall_survival_months"].max() * 1.02, 150))
    ax.legend(loc="lower left", fontsize=11, frameon=True, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Kaplan-Meier plot successfully saved to: {output_path}")


def plot_forest_plot(
    study_results: pd.DataFrame,
    meta_summary: dict,
    title: str = "Meta-Analysis Forest Plot: Cross-Cohort Model Generalization (AUC)",
    output_path: str = "forest_plot_corrected.png"
):
    """
    Plots a publication-quality Forest Plot displaying individual study AUCs,
    confidence intervals, study weights, and the pooled Meta-Analysis estimate (diamond).
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    study_labels = {
        "brca_tcga_pan_can_atlas_2018": "TCGA PanCancer Atlas",
        "brca_metabric": "METABRIC Cohort"
    }

    n_studies = len(study_results)
    y_positions = list(range(n_studies, 0, -1))

    # Calculate 95% CIs for individual studies
    study_names = []
    aucs = []
    ci_lowers = []
    ci_uppers = []
    sample_sizes = []

    for idx, row in study_results.iterrows():
        study_names.append(study_labels.get(row["test_study"], row["test_study"]))
        auc = row["auc"]
        se = row["auc_se"]
        aucs.append(auc)
        ci_lowers.append(max(0.0, auc - 1.96 * se))
        ci_uppers.append(min(1.0, auc + 1.96 * se))
        sample_sizes.append(int(row["n_samples"]))

    # Plot individual study point estimates and error bars
    for i, (y, auc, lower, upper, n) in enumerate(zip(y_positions, aucs, ci_lowers, ci_uppers, sample_sizes)):
        # Error bar (Confidence Interval)
        ax.plot([lower, upper], [y, y], color="#1f77b4", linewidth=2.5, solid_capstyle="round")
        # Square marker scaled by sample size
        box_size = np.sqrt(n) * 0.4
        ax.scatter(auc, y, s=box_size**2, color="#1f77b4", zorder=5, edgecolors="black", linewidth=1.2)
        # Annotation text
        label_text = f"{auc:.3f} [{lower:.3f}, {upper:.3f}] (N={n})"
        ax.text(1.05, y, label_text, va="center", ha="left", fontsize=10.5, fontfamily="monospace")

    # Pooled Random-Effects Result
    y_pooled = 0
    pooled_mean = meta_summary["random_effect_mean"]
    pooled_ci_low, pooled_ci_high = meta_summary["random_effect_ci95"]
    pooled_ci_low = max(0.0, pooled_ci_low)
    pooled_ci_high = min(1.0, pooled_ci_high)

    # Draw summary diamond for pooled result
    diamond_height = 0.35
    diamond_x = [pooled_ci_low, pooled_mean, pooled_ci_high, pooled_mean]
    diamond_y = [y_pooled, y_pooled + diamond_height/2, y_pooled, y_pooled - diamond_height/2]
    diamond = patches.Polygon(
        xy=list(zip(diamond_x, diamond_y)),
        closed=True,
        facecolor="#2ca02c",  # Green diamond for robust corrected meta-analysis
        edgecolor="black",
        linewidth=1.2,
        zorder=6
    )
    ax.add_patch(diamond)

    pooled_label = f"{pooled_mean:.3f} [{pooled_ci_low:.3f}, {pooled_ci_high:.3f}] (Pooled)"
    ax.text(1.05, y_pooled, pooled_label, va="center", ha="left", fontsize=10.5, fontweight="bold", fontfamily="monospace", color="#2ca02c")

    # Reference lines
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Chance Level (AUC = 0.50)")
    ax.axvline(pooled_mean, color="#2ca02c", linestyle=":", linewidth=1.5, alpha=0.7, label="Pooled Effect")

    # Y-axis ticks and labels
    all_y = y_positions + [y_pooled]
    all_labels = study_names + ["Random Effects Model (Pooled)"]
    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=11, fontweight="semibold")

    # Heterogeneity annotation
    het_text = (
        f"Heterogeneity: Cochran's Q = {meta_summary['cochran_q']:.2f} (p = {meta_summary['q_p_value']:.4f}), "
        f"$I^2$ = {meta_summary['i_squared_pct']:.1f}%, $\\tau^2$ = {meta_summary['tau_squared']:.4f}"
    )
    ax.text(
        0.5, -0.18,
        het_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#dddddd")
    )

    ax.set_xlim(0.3, 1.35)
    ax.set_xlabel("Area Under the ROC Curve (AUC)", fontsize=12, fontweight="semibold", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Forest plot successfully saved to: {output_path}")


def main():
    print("--- 1. Loading Clinical Data for Visualization ---")
    df = get_data("brca_clinical_data.csv")

    print("--- 2. Generating Kaplan-Meier Survival Curves ---")
    plot_kaplan_meier(df, output_path="kaplan_meier.png")

    print("--- 3. Running Corrected Biological Meta-Analysis Pipeline ---")
    pipeline = CancerMetaAnalysisPipeline(
        model_type="gradient_boosting",
        numeric_features=["age", "stage"],
        categorical_features=["er_status"]
    )
    study_results = pipeline.evaluate_leave_one_study_out(df, target_col="vital_status", study_col="study_id")
    meta_summary = CancerMetaAnalysisPipeline.pool_meta_analysis_results(study_results, metric="auc", se_col="auc_se")

    print("\nCorrected Study Results:")
    print(study_results.to_string(index=False))

    print("\nCorrected Meta-Analysis Summary:")
    for k, v in meta_summary.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        elif isinstance(v, tuple):
            print(f"  {k:25s}: [{v[0]:.4f}, {v[1]:.4f}]")
        else:
            print(f"  {k:25s}: {v}")

    print("\n--- 4. Generating Corrected Forest Plot (forest_plot_corrected.png) ---")
    plot_forest_plot(
        study_results,
        meta_summary,
        title="Corrected Meta-Analysis: Intrinsic Biological Features (Age, Stage, ER Status)",
        output_path="forest_plot_corrected.png"
    )


if __name__ == "__main__":
    main()
