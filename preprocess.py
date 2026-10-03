"""
preprocess.py
Clinical data loading, harmonization, and preprocessing pipeline for Breast Cancer Meta-Analysis.
Supports real cBioPortal/TCGA cohorts (brca_clinical_data.csv) and synthetic fallback.
"""

from typing import List, Dict, Tuple, Optional, Any
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class CohortHarmonizer:
    """
    Harmonizes clinical variables across multiple cancer cohorts / studies.
    Standardizes column names, stages, vital statuses, and cohort IDs.
    """

    @staticmethod
    def parse_stage(row: pd.Series) -> Optional[float]:
        """Maps AJCC staging or Nottingham Prognostic Index (NPI) to numeric tiers 1-4."""
        ajcc = str(row.get("AJCC_PATHOLOGIC_TUMOR_STAGE", "")).strip().upper()
        if "IV" in ajcc:
            return 4.0
        elif "III" in ajcc:
            return 3.0
        elif "II" in ajcc:
            return 2.0
        elif "I" in ajcc and "IV" not in ajcc:
            return 1.0

        # METABRIC Nottingham Prognostic Index fallback: <3.4 -> Stage 1, 3.4-5.4 -> Stage 2, >5.4 -> Stage 3
        try:
            npi = float(row.get("NPI", np.nan))
            if np.isnan(npi):
                return np.nan
            if npi < 3.4:
                return 1.0
            elif npi <= 5.4:
                return 2.0
            else:
                return 3.0
        except (ValueError, TypeError):
            return np.nan

    @staticmethod
    def parse_vital_status(val: Any) -> Optional[int]:
        """Converts diverse OS_STATUS formats ('0:LIVING', '1:DECEASED', 'Alive', 'Dead') to binary 0/1."""
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s.startswith("1") or "deceased" in s or "dead" in s or s == "event":
            return 1
        elif s.startswith("0") or "living" in s or "alive" in s or s == "censored":
            return 0
        return np.nan

    @classmethod
    def harmonize_brca_dataframe(cls, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonizes raw multi-cohort breast cancer dataframe (METABRIC + TCGA PanCancer).
        """
        df = df_raw.copy()

        # Standardize study identifier
        study_col = "studyId" if "studyId" in df.columns else "study_id"
        study_id = df[study_col].astype(str) if study_col in df.columns else "Study_Unknown"

        # Standardize patient identifier
        patient_col = "patientId" if "patientId" in df.columns else "patient_id"
        patient_id = df[patient_col].astype(str) if patient_col in df.columns else [f"PT_{i}" for i in range(len(df))]

        # Age harmonization
        age_tcga = pd.to_numeric(df.get("AGE"), errors="coerce")
        age_metabric = pd.to_numeric(df.get("AGE_AT_DIAGNOSIS"), errors="coerce")
        age = age_tcga.fillna(age_metabric)

        # Survival time (months)
        os_months = pd.to_numeric(df.get("OS_MONTHS"), errors="coerce")

        # Vital status
        status_raw = df.get("OS_STATUS") if "OS_STATUS" in df.columns else df.get("VITAL_STATUS")
        vital_status = status_raw.map(cls.parse_vital_status) if status_raw is not None else np.nan

        # Stage
        stage = df.apply(cls.parse_stage, axis=1)

        # Treatments
        chemo_raw = df.get("CHEMOTHERAPY", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
        chemotherapy = chemo_raw.map(lambda x: "Yes" if "YES" in x else ("No" if "NO" in x else "Unknown"))

        hormone_raw = df.get("HORMONE_THERAPY", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
        hormone_therapy = hormone_raw.map(lambda x: "Yes" if "YES" in x else ("No" if "NO" in x else "Unknown"))

        rad_raw = df.get("RADIATION_THERAPY", df.get("RADIO_THERAPY", pd.Series(index=df.index, dtype=str))).astype(str).str.upper()
        radiation_therapy = rad_raw.map(lambda x: "Yes" if "YES" in x else ("No" if "NO" in x else "Unknown"))

        # ER status
        er_source = df["ER_IHC"] if "ER_IHC" in df.columns else (df["SUBTYPE"] if "SUBTYPE" in df.columns else pd.Series(index=df.index, dtype=str))
        er_raw = er_source.fillna("Unknown").astype(str).str.lower()
        er_status = er_raw.map(lambda x: "Positive" if ("pos" in x or "luma" in x or "lumb" in x) else ("Negative" if ("neg" in x or "basal" in x) else "Unknown"))

        # Build clean dataframe
        clean_df = pd.DataFrame({
            "patient_id": patient_id,
            "study_id": study_id,
            "age": age,
            "stage": stage,
            "chemotherapy": chemotherapy,
            "hormone_therapy": hormone_therapy,
            "radiation_therapy": radiation_therapy,
            "er_status": er_status,
            "overall_survival_months": os_months,
            "vital_status": vital_status
        })

        # Keep patients with valid outcome and survival time
        valid_mask = clean_df["vital_status"].notnull() & clean_df["overall_survival_months"].notnull()
        clean_df = clean_df.loc[valid_mask].reset_index(drop=True)

        return clean_df


class StudyAwareStandardizer(BaseEstimator, TransformerMixin):
    """
    Standardizes continuous features within each study independently to reduce
    cohort-specific batch and center effects in cancer meta-analysis.
    """

    def __init__(self, study_col: str = "study_id"):
        self.study_col = study_col
        self.scalers_: Dict[str, StandardScaler] = {}
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns if col != self.study_col]
        self.feature_names_ = numeric_cols

        for study_id, group in X.groupby(self.study_col):
            scaler = StandardScaler()
            scaler.fit(group[numeric_cols])
            self.scalers_[study_id] = scaler
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        numeric_cols = self.feature_names_

        for study_id, group in X_out.groupby(self.study_col):
            if study_id in self.scalers_:
                scaler = self.scalers_[study_id]
                X_out.loc[group.index, numeric_cols] = scaler.transform(group[numeric_cols])
            else:
                fallback_scaler = StandardScaler()
                X_out.loc[group.index, numeric_cols] = fallback_scaler.fit_transform(group[numeric_cols])
        return X_out


def build_clinical_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer for imputing and encoding clinical variables.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def load_brca_clinical_data(csv_path: str = "brca_clinical_data.csv") -> pd.DataFrame:
    """
    Loads and cleans the Breast Cancer (BRCA) clinical dataset from CSV.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Clinical data file not found at: {csv_path}")

    print(f"Loading clinical data from {csv_path}...")
    df_raw = pd.read_csv(csv_path, low_memory=False)
    cleaned_df = CohortHarmonizer.harmonize_brca_dataframe(df_raw)
    print(f"Loaded and harmonized {len(cleaned_df)} patient records across cohorts:")
    for study, cnt in cleaned_df["study_id"].value_counts().items():
        print(f"  - {study}: {cnt} patients")
    return cleaned_df


def generate_synthetic_meta_cohorts(n_studies: int = 3, n_samples_per_study: int = 200) -> pd.DataFrame:
    """Fallback generator for testing when real data is unavailable."""
    np.random.seed(42)
    cohort_list = []

    for study_idx in range(1, n_studies + 1):
        study_name = f"Study_Cohort_{study_idx}"
        n = n_samples_per_study

        age = np.random.normal(60 + study_idx * 2, 10, size=n).clip(25, 90)
        stage = np.random.choice([1.0, 2.0, 3.0, 4.0], size=n, p=[0.3, 0.3, 0.25, 0.15])
        er_status = np.random.choice(["Positive", "Negative"], size=n, p=[0.7, 0.3])
        chemo = np.random.choice(["Yes", "No"], size=n, p=[0.4, 0.6])
        radiation = np.random.choice(["Yes", "No"], size=n, p=[0.6, 0.4])
        hormone = np.random.choice(["Yes", "No"], size=n, p=[0.65, 0.35])

        linear_risk = 0.03 * age + 0.4 * stage - 0.3 * (er_status == "Positive") + np.random.normal(0, 0.5, size=n)
        prob_event = 1 / (1 + np.exp(-linear_risk + 2.0))
        vital_status = np.random.binomial(1, prob_event)

        survival_months = np.random.exponential(scale=48, size=n) / (linear_risk - linear_risk.min() + 0.5)
        survival_months = survival_months.clip(1, 180)

        df_study = pd.DataFrame({
            "patient_id": [f"{study_name}_PT_{i:04d}" for i in range(n)],
            "study_id": study_name,
            "age": age,
            "stage": stage,
            "chemotherapy": chemo,
            "hormone_therapy": hormone,
            "radiation_therapy": radiation,
            "er_status": er_status,
            "overall_survival_months": survival_months,
            "vital_status": vital_status
        })

        cohort_list.append(df_study)

    return pd.concat(cohort_list, ignore_index=True)


def get_data(csv_path: str = "brca_clinical_data.csv") -> pd.DataFrame:
    """Helper that loads real data if present, else generates synthetic cohorts."""
    if os.path.exists(csv_path):
        return load_brca_clinical_data(csv_path)
    print(f"Notice: '{csv_path}' not found, falling back to synthetic data.")
    return generate_synthetic_meta_cohorts()


if __name__ == "__main__":
    print("Testing Preprocessing with real Breast Cancer clinical data...")
    data = get_data("brca_clinical_data.csv")
    print("\nDataset Summary:")
    print(data.info())
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nVital Status Distribution by Cohort:")
    print(pd.crosstab(data["study_id"], data["vital_status"], margins=True))
