"""
model.py
Machine Learning and Meta-Analysis Model Pipeline for Cancer Clinical Datasets.
Uses robust intrinsic biological/clinical features (Age, Stage, ER Status)
to avoid center-specific treatment documentation batch effects.
Includes:
- Binary outcome classification & clinical risk modeling
- Leave-One-Study-Out (LOSO) Cross-Validation across real clinical cohorts
- Meta-analytic aggregation (Fixed and Random Effects pooling with I^2 heterogeneity)
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score
from sklearn.pipeline import Pipeline

from preprocess import (
    build_clinical_preprocessor,
    get_data
)


class CancerMetaAnalysisPipeline:
    """
    Handles model training, validation across heterogeneous clinical cohorts,
    and meta-analytic pooling of evaluation metrics.
    """

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.random_state = random_state
        # Robust intrinsic biological features (excluding treatment documentation bias)
        self.numeric_features = numeric_features or ["age", "stage"]
        self.categorical_features = categorical_features or ["er_status"]
        self.pipeline = self._create_pipeline()

    def _get_classifier(self):
        if self.model_type == "logistic_regression":
            return LogisticRegression(penalty="l2", C=1.0, random_state=self.random_state, max_iter=1000)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.random_state)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _create_pipeline(self) -> Pipeline:
        preprocessor = build_clinical_preprocessor(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features
        )
        classifier = self._get_classifier()
        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits the pipeline on feature dataframe X and target y."""
        features_to_use = self.numeric_features + self.categorical_features
        self.pipeline.fit(X[features_to_use], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted class probabilities."""
        features_to_use = self.numeric_features + self.categorical_features
        return self.pipeline.predict_proba(X[features_to_use])[:, 1]

    def evaluate_leave_one_study_out(
        self,
        data: pd.DataFrame,
        target_col: str = "vital_status",
        study_col: str = "study_id"
    ) -> pd.DataFrame:
        """
        Performs Leave-One-Study-Out (LOSO) cross-validation across all studies.
        Trains on N-1 studies, evaluates on the held-out study.
        """
        unique_studies = data[study_col].unique()
        results = []

        features_to_use = self.numeric_features + self.categorical_features

        for test_study in unique_studies:
            train_mask = (data[study_col] != test_study)
            test_mask = (data[study_col] == test_study)

            X_train, y_train = data.loc[train_mask, features_to_use], data.loc[train_mask, target_col]
            X_test, y_test = data.loc[test_mask, features_to_use], data.loc[test_mask, target_col]

            # Clone and fit pipeline
            pipeline = self._create_pipeline()
            pipeline.fit(X_train, y_train)

            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan
            pr_auc = average_precision_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan
            brier = brier_score_loss(y_test, y_pred_proba)
            acc = accuracy_score(y_test, y_pred)
            n_samples = len(y_test)

            # Standard error approximation for AUC (Hanley-McNeil formulation)
            q1 = auc / (2 - auc)
            q2 = (2 * auc**2) / (1 + auc)
            n1 = np.sum(y_test == 1)
            n0 = np.sum(y_test == 0)
            if n1 > 0 and n0 > 0:
                auc_var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0)
                auc_se = np.sqrt(max(auc_var, 1e-6))
            else:
                auc_se = np.nan

            results.append({
                "test_study": test_study,
                "n_samples": n_samples,
                "n_events": int(n1),
                "auc": auc,
                "auc_se": auc_se,
                "pr_auc": pr_auc,
                "brier_score": brier,
                "accuracy": acc
            })

        return pd.DataFrame(results)

    @staticmethod
    def pool_meta_analysis_results(study_results_df: pd.DataFrame, metric: str = "auc", se_col: str = "auc_se") -> Dict[str, Any]:
        """
        Computes Fixed Effects and Random Effects (DerSimonian-Laird) meta-analysis pooling
        along with Cochran's Q and I^2 heterogeneity metrics.
        """
        valid_df = study_results_df.dropna(subset=[metric, se_col]).copy()
        k = len(valid_df)

        if k < 2:
            raise ValueError("Meta-analysis requires at least 2 valid study estimates.")

        y_i = valid_df[metric].values
        w_i = 1.0 / (valid_df[se_col].values ** 2)

        # Fixed-effect pooled estimate
        fixed_estimate = np.sum(w_i * y_i) / np.sum(w_i)
        fixed_se = np.sqrt(1.0 / np.sum(w_i))

        # Cochran's Q test for heterogeneity
        q = np.sum(w_i * (y_i - fixed_estimate) ** 2)
        df_q = k - 1
        q_p_value = 1.0 - stats.chi2.cdf(q, df=df_q)

        # I^2 index of heterogeneity
        i2 = max(0.0, ((q - df_q) / q) * 100) if q > 0 else 0.0

        # DerSimonian-Laird between-study variance (tau^2)
        c = np.sum(w_i) - (np.sum(w_i ** 2) / np.sum(w_i))
        tau2 = max(0.0, (q - df_q) / c) if c > 0 else 0.0

        # Random-effects pooled estimate
        w_re_i = 1.0 / (valid_df[se_col].values ** 2 + tau2)
        random_estimate = np.sum(w_re_i * y_i) / np.sum(w_re_i)
        random_se = np.sqrt(1.0 / np.sum(w_re_i))

        # 95% Confidence intervals
        z_crit = 1.96
        return {
            "num_studies": k,
            "metric": metric,
            "fixed_effect_mean": float(fixed_estimate),
            "fixed_effect_ci95": (float(fixed_estimate - z_crit * fixed_se), float(fixed_estimate + z_crit * fixed_se)),
            "random_effect_mean": float(random_estimate),
            "random_effect_ci95": (float(random_estimate - z_crit * random_se), float(random_effect_mean := random_estimate + z_crit * random_se)),
            "cochran_q": float(q),
            "q_p_value": float(q_p_value),
            "i_squared_pct": float(i2),
            "tau_squared": float(tau2)
        }


if __name__ == "__main__":
    print("--- 1. Loading Real Multi-Cohort Breast Cancer Clinical Dataset ---")
    data = get_data("brca_clinical_data.csv")
    print(f"Total samples: {len(data)} across {data['study_id'].nunique()} cohorts.")

    print("\n--- 2. Running Leave-One-Study-Out (LOSO) Cross-Validation (Biological Features) ---")
    pipeline = CancerMetaAnalysisPipeline(
        model_type="gradient_boosting",
        numeric_features=["age", "stage"],
        categorical_features=["er_status"]
    )
    study_results = pipeline.evaluate_leave_one_study_out(data, target_col="vital_status", study_col="study_id")
    print(study_results.to_string(index=False))

    print("\n--- 3. Meta-Analytic Metric Pooling (Corrected AUC Meta-Analysis) ---")
    meta_summary = CancerMetaAnalysisPipeline.pool_meta_analysis_results(study_results, metric="auc", se_col="auc_se")
    for k, v in meta_summary.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.4f}")
        elif isinstance(v, tuple):
            print(f"{k:25s}: [{v[0]:.4f}, {v[1]:.4f}]")
        else:
            print(f"{k:25s}: {v}")
