"""
fetch_data.py
Downloads real public Breast Cancer (BRCA) clinical datasets from the cBioPortal REST API.
Fetches multi-cohort datasets (e.g., TCGA Pan-Cancer & METABRIC) for cancer meta-analysis.
"""

import sys
import requests
import pandas as pd


CBIOPORTAL_API_URL = "https://www.cbioportal.org/api"

# Public breast cancer studies with rich clinical and survival annotations
STUDIES = [
    "brca_tcga_pan_can_atlas_2018",
    "brca_metabric"
]


def fetch_study_clinical_data(study_id: str) -> pd.DataFrame:
    """
    Fetches raw clinical records for a study from cBioPortal and pivots to a wide tabular format.
    """
    url = f"{CBIOPORTAL_API_URL}/studies/{study_id}/clinical-data"
    params = {
        "clinicalDataType": "PATIENT",
        "projection": "DETAILED"
    }
    headers = {
        "Accept": "application/json",
        "User-Agent": "AntigravityCancerMetaAnalysis/1.0"
    }

    print(f"Fetching clinical data for study: {study_id} ...")
    response = requests.get(url, params=params, headers=headers, timeout=60)
    
    if response.status_code != 200:
        # Fallback without parameters
        print(f"Retrying without clinicalDataType filter for {study_id}...")
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

    records = response.json()
    if not records:
        print(f"Warning: No records found for study {study_id}")
        return pd.DataFrame()

    df_raw = pd.DataFrame(records)
    print(f"Retrieved {len(df_raw)} clinical attribute entries for {study_id}.")

    # Pivot long format (patientId, clinicalAttributeId, value) to wide format
    pivot_cols = ["patientId"]
    if "studyId" in df_raw.columns:
        pivot_cols.append("studyId")

    df_wide = df_raw.pivot_table(
        index=pivot_cols,
        columns="clinicalAttributeId",
        values="value",
        aggfunc="first"
    ).reset_index()

    if "studyId" not in df_wide.columns:
        df_wide["studyId"] = study_id

    return df_wide


def main(output_file: str = "brca_clinical_data.csv"):
    cohort_dfs = []
    for study in STUDIES:
        try:
            df = fetch_study_clinical_data(study)
            if not df.empty:
                cohort_dfs.append(df)
        except Exception as e:
            print(f"Error fetching study {study}: {e}")

    if not cohort_dfs:
        print("Failed to fetch data from remote API.", file=sys.stderr)
        sys.exit(1)

    combined_df = pd.concat(cohort_dfs, ignore_index=True)
    print(f"\nCombined clinical dataset shape: {combined_df.shape}")
    print(f"Studies fetched: {combined_df['studyId'].value_counts().to_dict()}")

    combined_df.to_csv(output_file, index=False)
    print(f"Saved clinical dataset successfully to: {output_file}")


if __name__ == "__main__":
    output_path = "brca_clinical_data.csv"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    main(output_path)
