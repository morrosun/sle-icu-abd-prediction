"""SLECI 01: baseline characteristics (Table 1) by CAM-ICU delirium status."""
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
df = pd.read_csv(os.path.join(BASE, "data", "mimic_sle_analytic.csv"))

# Primary analysis set: patients with at least one CAM-ICU assessment
ana = df[df.cam_assessed == 1].copy()
ana["grp"] = np.where(ana.cam_pos == 1, "Delirium", "No delirium")

CONT = [
    ("age", "Age, years"), ("bmi", "BMI, kg/m2"),
    ("sofa_24h", "SOFA (24h)"), ("apsiii", "APS III"), ("sapsii", "SAPS II"),
    ("oasis", "OASIS"), ("charlson", "Charlson index"),
    ("gcs_min_24h", "GCS min (24h)"),
    ("heart_rate_mean", "Heart rate, mean"), ("sbp_mean", "SBP, mean"),
    ("mbp_min", "MAP, min"), ("resp_rate_mean", "Resp rate, mean"),
    ("temperature_max", "Temperature, max"), ("spo2_min", "SpO2, min"),
    ("wbc_max", "WBC, max"), ("hemoglobin_min", "Hemoglobin, min"),
    ("platelets_min", "Platelets, min"), ("albumin_min", "Albumin, min"),
    ("creatinine_max", "Creatinine, max"), ("bun_max", "BUN, max"),
    ("sodium_min", "Sodium, min"), ("sodium_max", "Sodium, max"),
    ("bicarbonate_min", "Bicarbonate, min"), ("aniongap_max", "Anion gap, max"),
    ("glucose_max", "Glucose, max"), ("inr_max", "INR, max"),
    ("bilirubin_total_max", "Bilirubin, max"),
    ("abs_lymphocytes_min", "Lymphocytes, min"), ("abs_neutrophils_max", "Neutrophils, max"),
    ("lactate_max", "Lactate, max"), ("ph_min", "pH, min"),
    ("pao2fio2ratio_min", "PaO2/FiO2, min"), ("uo_24h", "Urine output 24h"),
    ("pred_eq_24h", "Prednisone-eq 24h, mg"),
    ("los_icu", "ICU LOS, days"), ("los_hospital", "Hospital LOS, days"),
]
CAT = [
    ("female", "Female"), ("lupus_nephritis_icd", "Lupus nephritis (ICD)"),
    ("sle_organ_involv", "SLE organ involvement (ICD)"),
    ("primary_brain_injury", "Primary acute brain injury"),
    ("htn", "Hypertension"), ("diabetes", "Diabetes"), ("ckd", "CKD"),
    ("esrd_dialysis", "ESRD/dialysis"), ("epilepsy_hx", "Epilepsy history"),
    ("dementia_hx", "Dementia"), ("depression_hx", "Depression/mood disorder"),
    ("alcohol_abuse", "Alcohol abuse"), ("aps_antiphospholipid", "Antiphospholipid syndrome"),
    ("chf", "Heart failure"), ("copd", "COPD"), ("malignancy", "Malignancy"),
    ("sepsis3", "Sepsis-3"), ("mech_vent", "Mechanical ventilation"),
    ("vaso_24h", "Vasoactive agent (24h)"), ("rrt_24h", "RRT (24h)"),
    ("steroid_hosp_any", "Any corticosteroid (hospital)"),
    ("steroid_24h", "Corticosteroid in ICU 24h"),
    ("pulse_steroid", "Pulse steroid (>=250mg)"),
    ("hcq", "Hydroxychloroquine"), ("mmf", "Mycophenolate"), ("aza", "Azathioprine"),
    ("cyc", "Cyclophosphamide"), ("cni", "Calcineurin inhibitor"),
    ("benzo_24h", "Benzodiazepine (24h)"), ("propofol_24h", "Propofol (24h)"),
    ("dexmed_24h", "Dexmedetomidine (24h)"), ("opioid_24h", "Opioid (24h)"),
    ("haloperidol", "Haloperidol"), ("atypical_ap", "Atypical antipsychotic"),
    ("icd_seizure", "Seizure (ICD)"), ("icd_psychosis", "Psychosis (ICD)"),
    ("hospital_expire_flag", "In-hospital mortality"), ("mort_28d", "28-day mortality"),
    ("mort_90d", "90-day mortality"),
]

g1 = ana[ana.grp == "Delirium"]
g0 = ana[ana.grp == "No delirium"]
rows = []


def fmt_cont(s):
    s = s.dropna()
    if len(s) == 0:
        return "-"
    return f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})"


rows.append({"Variable": "N", "Overall": len(ana), "No delirium": len(g0),
             "Delirium": len(g1), "P": "", "Missing %": ""})

for col, lab in CONT:
    if col not in ana.columns:
        continue
    a, b = g1[col].dropna(), g0[col].dropna()
    p = stats.mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan
    rows.append({
        "Variable": f"{lab}, median (IQR)",
        "Overall": fmt_cont(ana[col]), "No delirium": fmt_cont(g0[col]),
        "Delirium": fmt_cont(g1[col]),
        "P": f"{p:.3f}" if pd.notna(p) else "-",
        "Missing %": f"{100*ana[col].isna().mean():.1f}",
    })

for col, lab in CAT:
    if col not in ana.columns:
        continue
    tab = pd.crosstab(ana[col].fillna(0), ana.grp)
    try:
        p = stats.chi2_contingency(tab)[1] if tab.shape == (2, 2) else np.nan
    except Exception:
        p = np.nan
    def pct(s):
        s = s[col].fillna(0)
        return f"{int(s.sum())} ({100*s.mean():.1f})"
    rows.append({
        "Variable": f"{lab}, n (%)",
        "Overall": pct(ana), "No delirium": pct(g0), "Delirium": pct(g1),
        "P": f"{p:.3f}" if pd.notna(p) else "-",
        "Missing %": f"{100*ana[col].isna().mean():.1f}",
    })

t1 = pd.DataFrame(rows)
t1.to_csv(rf"{BASE}\output\table1_camicu.csv", index=False, encoding="utf-8-sig")

with open(rf"{BASE}\output\table1_camicu.md", "w", encoding="utf-8") as f:
    f.write("# Table 1. Baseline characteristics by ICU delirium (CAM-ICU) status\n\n")
    f.write(f"Analysis set: SLE ICU admissions with >=1 CAM-ICU assessment (N={len(ana)}), "
            f"of {len(df)} eligible admissions.\n\n")
    f.write(t1.to_markdown(index=False))

print(f"analysis set N={len(ana)}  delirium={len(g1)} ({100*len(g1)/len(ana):.1f}%)  "
      f"no delirium={len(g0)}")
print(t1.head(45).to_string(index=False))
