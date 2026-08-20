# SLE-ICU Acute Brain Dysfunction Prediction (SLECI)

Analysis code for the study:

> **Early benzodiazepine exposure and acute brain dysfunction in critically ill patients with systemic lupus erythematosus: a cross-database machine learning study** (Chinese submission, *Chinese Journal of Rheumatology*, manuscript version v16).

Retrospective, multicenter cohort study identifying modifiable risk factors and building/validating prediction models for acute brain dysfunction (ABD) in critically ill SLE patients:

- **Development cohort**: MIMIC-IV (N = 264; outcome = CAM-ICU positive delirium, 105 events)
- **External validation**: eICU-CRD (N = 256 for the diagnosis-only endpoint; N = 202 with GCS records)
- **Models**: Logistic regression, LASSO-LR, Random Forest, XGBoost; 10-fold stratified cross-validation; external validation; logit recalibration; internal–external cross-validation (IECV); SHAP.
- **Key sensitivity analyses**: removal of GCS (circularity), diagnosis-only endpoint, pre-specified 9-variable parsimonious core model (EPV ≈ 11.7) with Firth penalized logistic regression, reverse-causality landmark analysis, NWICU positive control.

## Repository layout

```
sql/    PostgreSQL data-extraction queries (MIMIC-IV & eICU-CRD, local PhysioNet deployment)
code/   Python analysis scripts (baseline table, regression, RCS, ML, sensitivities, Firth, IECV)
```

## Pipeline order

1. **SQL extraction** (`sql/`): feasibility (01–02), cohort construction (10–16), eICU cohort (20–21), NWICU transfer (24), benzodiazepine timing (26/26b/26c). Produces analytic datasets under `data/` (not versioned — see Data statement).
2. **Python analysis** (`code/`), run in order:
   - `01_table1.py` — baseline characteristics (Table 1)
   - `02_regression.py` — multivariable logistic regression, RCS, subgroups
   - `03_rcs_subgroup.py` — restricted cubic splines / subgroup analyses
   - `04_ml_model.py` — main ML pipeline (LASSO selection, 4 models, 10-fold CV, external validation, calibration, DCA, SHAP)
   - `05_ml_sens_nogcs.py` — sensitivity A/B (remove GCS predictor)
   - `06_recal_expand.py` — expanded diagnosis-only external cohort + logit recalibration
   - `07_eicu_split_validation.py` — eICU internal–external cross-validation (IECV)
   - `08_nwicu_transfer.py` — NWICU positive-control transportability
   - `09_core_firth.py` — pre-specified 9-variable core model + Firth penalized logistic (EPV)
   - `optimism.py` — apparent vs cross-validated AUC optimism
   - `landmark_benzo.py` — reverse-causality landmark analysis (requires direct DB access)
   - `table1_combined.py` — combined DEV/VAL Table 1

## Requirements

- Python ≥ 3.12 with: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `statsmodels`, `shap`
- PostgreSQL 18 with locally deployed MIMIC-IV and eICU-CRD schemas (`mimiciv`, `eicu_crd`), plus `nwicu` for the positive-control analysis
- `psycopg2` (only for `landmark_benzo.py`)

## Data statement (PhysioNet DUA)

The raw data (MIMIC-IV, eICU-CRD, NWICU) are governed by the PhysioNet Credentialed Health Data Use Agreement and are **not** redistributed here. Scripts read analytic datasets from `data/` (produced by the SQL queries against a local deployment); `data/` is excluded from version control. The user must have a PhysioNet credential and a local database deployment to reproduce the analysis.

Database credentials are read from environment variables: `PGHOST`, `PGUSER`, `PGPASSWORD`, `PGDATABASE` (see `code/landmark_benzo.py`).

## Outputs

All results are written to `output/` (excluded from version control; they are reproducible from the code and data).

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code in your research, please cite the Zenodo record:/n/n> Wang, Kai. (2026). sle-icu-abd-prediction: analysis code for early benzodiazepine exposure and acute brain dysfunction in critically ill SLE patients (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.22022351

Version DOI: `10.5281/zenodo.22022351` · Concept DOI: `10.5281/zenodo.22022350` · Repository: https://github.com/morrosun/sle-icu-abd-prediction
