# Cross-Cohort Breast Cancer Survival Meta-Analysis Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![cBioPortal API](https://img.shields.io/badge/Data-cBioPortal%20REST%20API-orange.svg)](https://www.cbioportal.org/)

An end-to-end, reproducible Machine Learning and Meta-Analysis pipeline designed to evaluate cross-cohort generalization and eliminate center-specific batch effects in clinical oncology datasets.

---

## 🔬 Abstract

Clinical risk and survival models frequently suffer from significant performance degradation when deployed across external hospital centers and genomic cohorts. This repository implements a robust cancer meta-analysis framework applied to **3,065 Breast Invasive Carcinoma (BRCA) patients** across two major public multicenter cohorts: **TCGA PanCancer Atlas** ($N=1,084$) and **METABRIC** ($N=1,981$).

Using **Leave-One-Study-Out (LOSO) Cross-Validation** coupled with **DerSimonian-Laird Random-Effects Meta-Analysis**, this pipeline:
1. Identifies center-level batch effects stemming from clinical documentation discrepancies (e.g., radiotherapy annotations causing an initial $I^2 = 97.2\%$ heterogeneity).
2. Isolates intrinsic biological features (Age, Tumor Stage, ER status), reducing between-study variance to **$I^2 = 0.0\%$** and boosting held-out external generalization AUC to **$0.647$ (95% CI: $[0.626, 0.669]$)**.

---

## 📐 Pipeline Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │     cBioPortal Public REST API Ingestion     │
                    │   (TCGA PanCancer Atlas + METABRIC Cohorts)  │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │           Clinical Harmonization             │
                    │  - Staging: AJCC Pathological & NPI mapping  │
                    │  - Vital Status: Standardized 0/1 encoding   │
                    │  - Missing Value Imputation & Encoding       │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │    Leave-One-Study-Out (LOSO) Validation     │
                    │  - Train: Study A -> Test: Study B           │
                    │  - Train: Study B -> Test: Study A           │
                    └──────────────────────┬───────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │                                           │
                     ▼                                           ▼
┌─────────────────────────────────────────┐ ┌─────────────────────────────────────────┐
│     Batch Effect Diagnostic & Audit     │ │    Meta-Analytic Statistical Pooling    │
│  - 5-Fold Permutation Importance        │ │  - DerSimonian-Laird Random Effects     │
│  - Discovered Treatment Doc Bias        │ │  - Cochran's Q Test & I² Heterogeneity  │
└─────────────────────────────────────────┘ └─────────────────────────────────────────┘
```

---

## 📊 Key Meta-Analysis Results

| Pipeline Stage | Evaluated Features | METABRIC Held-Out AUC | TCGA Held-Out AUC | Pooled Meta-AUC (95% CI) | Heterogeneity ($I^2$) | Cochran's $Q$ ($p$-value) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Initial (Uncorrected)** | Biological + Treatment Documentation | `0.467` | `0.639` | `0.552` [$0.384, 0.720$] | **`97.2%`** | $35.26$ ($p < 0.0001$) |
| **Corrected (Biological Only)** | Age, Stage, ER Status | **`0.650`** | **`0.634`** | **`0.647` [$0.626, 0.669$]** | **`0.0%`** | **$0.32$ ($p = 0.5723$)** |

### Top Clinical Features (Permutation Impact on ROC-AUC)
1. **Radiation Therapy Documentation** (40.4% relative impact — *Identified as primary center-specific batch effect artifact*)
2. **Age at Diagnosis** (34.2% relative impact — *Robust biological risk factor*)
3. **Tumor Stage / Nottingham Prognostic Index** (13.9% relative impact — *Standard anatomical predictor*)
4. **ER Receptor Status** (8.7% relative impact — *Key molecular subtype determinant*)

---

## 📁 Repository Structure

```
├── requirements.txt            # Python dependencies (pip compatible with Python 3.10 - 3.14)
├── fetch_data.py               # Downloads and pivots clinical data from cBioPortal REST API
├── preprocess.py              # Cohort harmonization, staging logic, and scikit-learn transformers
├── model.py                   # LOSO cross-validation and DerSimonian-Laird meta-analytic pooling
├── feature_importance.py      # Permutation feature importance and batch effect diagnostics
├── visualize.py               # Generates publication-ready Kaplan-Meier and Forest Plots
├── brca_clinical_data.csv     # Combined harmonized clinical dataset (3,593 records)
├── kaplan_meier.png           # Overall survival curves with Log-Rank test (300 DPI)
├── forest_plot.png            # Initial meta-analysis forest plot (300 DPI)
├── forest_plot_corrected.png  # Corrected biological meta-analysis forest plot (300 DPI)
└── feature_importance.png     # Feature importance bar chart across CV folds (300 DPI)
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/<your-username>/cancer-meta-analysis.git
cd cancer-meta-analysis
pip install -r requirements.txt
```

### 2. Step-by-Step Execution Workflow

#### Step 1: Download Real Multi-Cohort Clinical Data
Fetches public TCGA PanCancer and METABRIC cohorts from the cBioPortal REST API:
```bash
python fetch_data.py
```

#### Step 2: Validate Data Harmonization
Preprocesses variables, standardizes TNM stages/NPI, and handles missing values:
```bash
python preprocess.py
```

#### Step 3: Run the Machine Learning & Meta-Analysis Pipeline
Executes Leave-One-Study-Out cross-validation and computes fixed/random-effects pooled estimates:
```bash
python model.py
```

#### Step 4: Run Feature Importance & Batch Effect Diagnostics
Calculates model feature importances and 5-fold cross-validated permutation scores:
```bash
python feature_importance.py
```

#### Step 5: Generate Publication-Ready Visualizations
Renders Kaplan-Meier survival curves and Forest Plots at 300 DPI:
```bash
python visualize.py
```

---

## 📈 Visualizations

- **Kaplan-Meier Survival Analysis (`kaplan_meier.png`)**: Compares overall survival between TCGA and METABRIC cohorts with confidence bands and Log-Rank test statistics ($\chi^2 = 0.69, p = 0.407$).
- **Corrected Forest Plot (`forest_plot_corrected.png`)**: Illustrates consistent cross-cohort generalizability ($0.634$ and $0.650$) and zero residual heterogeneity ($I^2 = 0.0\%$).
- **Feature Importance Analysis (`feature_importance.png`)**: Highlights clinical attribute rankings by permutation impact on ROC-AUC.

---

## 🛠️ Methodological Details

### Harmonization Logic
- **Staging**: Mapped AJCC TNM pathologic stages (Stages I–IV) and METABRIC Nottingham Prognostic Index (NPI) into standardized tiers:
  - $\text{NPI} < 3.4 \implies \text{Stage 1}$
  - $3.4 \le \text{NPI} \le 5.4 \implies \text{Stage 2}$
  - $\text{NPI} > 5.4 \implies \text{Stage 3}$
- **Vital Status**: Uniformly mapped to binary indicators ($0 = \text{Living/Censored}$, $1 = \text{Deceased}$).

### Meta-Analytic Estimators
- **Standard Error Approximation for AUC**: Hanley-McNeil formulation:
  $$\text{Var}(\text{AUC}) = \frac{\text{AUC}(1-\text{AUC}) + (n_1 - 1)(Q_1 - \text{AUC}^2) + (n_0 - 1)(Q_2 - \text{AUC}^2)}{n_1 n_0}$$
- **Random-Effects Pooling**: DerSimonian-Laird inverse-variance weighting with between-study variance estimator $\tau^2$ and Cochran's $Q$ heterogeneity statistic.

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
