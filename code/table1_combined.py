# -*- coding: utf-8 -*-
"""SLECI: faithful combined Table 1 (DEV=MIMIC N=264, VAL=eICU N=202).
Mirrors the exact model cohort filters; uses median(IQR)+Mann-Whitney for
continuous (skewed) vars and n(%)+chi2 for categorical, with P columns.
Regenerates the manuscript Table 1 that was previously hand-built (mean+-SD,
no P, wrong N)."""
import os, numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = pd.read_csv(os.path.join(ROOT, "data", "mimic_sle_analytic.csv"))
e = pd.read_csv(os.path.join(ROOT, "data", "eicu_sle_analytic.csv"))

# ---- exact model cohort filters ----
dev = m[(m.cam_assessed == 1) & (m.primary_brain_injury == 0)].copy()
dev["sofa_noncns"] = dev["sofa_24h"] - dev["sofa_cns"]
dev["y"] = dev["cam_pos"].astype(int)
val = e[(e.primary_brain_injury == 0) & (e.gcs_min_24h.notna())].copy()
if "sofa_noncns" not in val.columns:
    val["sofa_noncns"] = val["sofa_24h"] - val["sofa_cns"]
val["y"] = val["abd_dx_gcs"].astype(int)

CONT = ["age", "sofa_noncns", "gcs_min_24h", "bicarbonate_min", "lactate_max",
        "creatinine_max", "bun_max", "wbc_max", "platelets_min", "hemoglobin_min",
        "glucose_max", "sodium_min", "temperature_max", "heart_rate_mean",
        "resp_rate_mean", "mbp_min", "los_icu", "los_hospital"]
CAT = ["female", "sepsis3_24h", "mech_vent_24h", "benzo_24h", "opioid_24h",
       "steroid_24h", "vaso_24h", "propofol_24h", "dexmed_24h", "htn",
       "diabetes", "ckd", "esrd_dialysis", "epilepsy_hx", "chf", "malignancy",
       "lupus_nephritis_icd", "hcq"]

LAB = {"age": "Age, years", "sofa_noncns": "Non-CNS SOFA", "gcs_min_24h": "GCS min (24h)",
       "bicarbonate_min": "Bicarbonate min", "lactate_max": "Lactate max",
       "creatinine_max": "Creatinine max", "bun_max": "BUN max", "wbc_max": "WBC max",
       "platelets_min": "Platelets min", "hemoglobin_min": "Hemoglobin min",
       "glucose_max": "Glucose max", "sodium_min": "Sodium min",
       "temperature_max": "Temperature max", "heart_rate_mean": "Heart rate mean",
       "resp_rate_mean": "Resp rate mean", "mbp_min": "MAP min",
       "los_icu": "ICU LOS, days", "los_hospital": "Hospital LOS, days",
       "female": "Female", "sepsis3_24h": "Sepsis-3", "mech_vent_24h": "Mechanical ventilation",
       "benzo_24h": "Benzodiazepine", "opioid_24h": "Opioid", "steroid_24h": "Corticosteroid",
       "vaso_24h": "Vasoactive agent", "propofol_24h": "Propofol",
       "dexmed_24h": "Dexmedetomidine", "htn": "Hypertension", "diabetes": "Diabetes",
       "ckd": "CKD", "esrd_dialysis": "ESRD/dialysis", "epilepsy_hx": "Epilepsy history",
       "chf": "Heart failure", "malignancy": "Malignancy",
       "lupus_nephritis_icd": "Lupus nephritis", "hcq": "Hydroxychloroquine"}

def block(d, ycol):
    g1 = d[d[ycol] == 1]; g0 = d[d[ycol] == 0]
    rows = []
    for c in CONT:
        if c not in d.columns:
            continue
        a, b = g1[c].dropna(), g0[c].dropna()
        p = stats.mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan
        def med(s):
            s = s.dropna()
            return f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})"
        rows.append([LAB.get(c, c), med(d[c]), med(g0[c]), med(g1[c]),
                     f"{p:.3f}" if pd.notna(p) else "-",
                     f"{100*d[c].isna().mean():.1f}"])
    for c in CAT:
        if c not in d.columns:
            continue
        tab = pd.crosstab(d[c].fillna(0), d[ycol])
        try:
            p = stats.chi2_contingency(tab)[1] if tab.shape == (2, 2) else np.nan
        except Exception:
            p = np.nan
        def pct(s):
            s = s[c].fillna(0)
            return f"{int(s.sum())} ({100*s.mean():.1f})"
        rows.append([LAB.get(c, c), pct(d), pct(g0), pct(g1),
                     f"{p:.3f}" if pd.notna(p) else "-",
                     f"{100*d[c].isna().mean():.1f}"])
    return rows

rd = block(dev, "y"); rv = block(val, "y")
# align variable order by union
order = [r[0] for r in rd]
rv_map = {r[0]: r for r in rv}
out = []
out.append(["Variable", "DEV Overall", "DEV No-ABD", "DEV ABD", "DEV P",
            "VAL Overall", "VAL No-ABD", "VAL ABD", "VAL P", "Missing%"])
for r in rd:
    name = r[0]
    v = rv_map.get(name, [name, "-", "-", "-", "-", "-"])
    out.append([name, r[1], r[2], r[3], r[4], v[1], v[2], v[3], v[4], r[5]])
# append VAL-only vars
for r in rv:
    if r[0] not in order:
        out.append([r[0], "-", "-", "-", "-", r[1], r[2], r[3], r[4], r[5]])

t = pd.DataFrame(out[1:], columns=out[0])
print(f"DEV N={len(dev)} events={int(dev.y.sum())} | VAL N={len(val)} events={int(val.y.sum())}")
print(t.to_string(index=False))
t.to_csv(os.path.join(ROOT, "output", "table1_revised.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(ROOT, "output", "table1_revised.md"), "w", encoding="utf-8") as f:
    f.write(f"# Table 1 (revised). Baseline characteristics by ABD status\n\n")
    f.write(f"DEV=MIMIC-IV (N={len(dev)}, events={int(dev.y.sum())}); "
            f"VAL=eICU-CRD (N={len(val)}, events={int(val.y.sum())}).\n")
    f.write("Continuous: median (IQR), Mann-Whitney U; Categorical: n (%), chi-square.\n\n")
    f.write(t.to_markdown(index=False))
