# -*- coding: utf-8 -*-
# DB credentials come from environment variables (PGHOST/PGUSER/PGPASSWORD/PGDATABASE);
# data are from a local PhysioNet deployment, see README for details.
"""SLECI: reverse-causality landmark / early-window sensitivity for benzodiazepine.
DEV = MIMIC-IV (N=264, cam_assessed & no primary brain injury), outcome = cam_pos.
Queries first benzo hour from mimiciv_icu.inputevents directly via psycopg2, then
compares adjusted OR of the original 24h-window exposure vs an early-window
(first dose <=12h) exposure. If agitated delirium -> sedation dominated, early
(pre-delirium) exposure would show a weaker association.
"""
import os, numpy as np, pandas as pd
import psycopg2, statsmodels.api as sm
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = pd.read_csv(os.path.join(ROOT, "data", "mimic_sle_analytic.csv"))
dev = m[(m.cam_assessed == 1) & (m.primary_brain_injury == 0)].copy().reset_index(drop=True)
dev["sofa_noncns"] = dev["sofa_24h"] - dev["sofa_cns"]
dev["y"] = dev["cam_pos"].astype(int)
ids = dev.stay_id.astype(int).tolist()
vals = ",".join(f"({i})" for i in ids)

conn = psycopg2.connect(host=os.environ.get("PGHOST", "localhost"),
                         user=os.environ.get("PGUSER", "postgres"),
                         password=os.environ.get("PGPASSWORD", ""),
                         dbname=os.environ.get("PGDATABASE", "mimiciv"))
q = f"""
with dev_stay(stay_id) as (values {vals})
select ds.stay_id,
       min(extract(epoch from ie.starttime - ic.intime)/3600.0) as benzo_first_hour
from dev_stay ds
join mimiciv_icu.icustays ic on ic.stay_id = ds.stay_id
join mimiciv_icu.inputevents ie on ie.stay_id = ds.stay_id
join mimiciv_icu.d_items di on di.itemid = ie.itemid
where lower(di.label) ~ 'midazolam|lorazepam|diazepam'
  and ie.starttime >= ic.intime
  and ie.starttime <= ic.intime + interval '24 hour'
group by ds.stay_id
"""
tim = pd.read_sql(q, conn); conn.close()
tim["benzo_first_hour"] = tim["benzo_first_hour"].astype(float)
dev = dev.merge(tim, on="stay_id", how="left")
dev["benzo_early12h"] = ((dev.benzo_24h == 1) & (dev.benzo_first_hour <= 12)).astype(int)
dev["benzo_late12h"]  = ((dev.benzo_24h == 1) & (dev.benzo_first_hour > 12)).astype(int)

COV = ["age", "female", "sofa_noncns", "gcs_min_24h", "bicarbonate_min",
       "sepsis3_24h", "mech_vent_24h", "opioid_24h", "steroid_24h"]
for c in COV + ["benzo_24h", "benzo_early12h", "benzo_late12h"]:
    dev[c] = dev[c].fillna(dev[c].median())

def aor(df, exp):
    X = sm.add_constant(df[COV + [exp]])
    res = sm.Logit(df["y"], X).fit(disp=0)
    coef = res.params[exp]; ci = res.conf_int().loc[exp]
    return np.exp(coef), np.exp(ci[0]), np.exp(ci[1]), int(res.nobs)

rows = []
for exp, lab in [("benzo_24h", "Benzodiazepine (any, 24h)"),
                 ("benzo_early12h", "Benzodiazepine (first dose <=12h)"),
                 ("benzo_late12h", "Benzodiazepine (first dose >12h)")]:
    orr, lo, hi, n = aor(dev, exp)
    rows.append([lab, int(dev[exp].sum()), f"{orr:.2f}", f"{lo:.2f}-{hi:.2f}"])
    print(f"{lab:42s} exposed={int(dev[exp].sum()):3d}  aOR={orr:.2f} (95%CI {lo:.2f}-{hi:.2f})  n={n}")

exp = dev[dev.benzo_24h == 1]
h1 = exp[exp.y == 1].benzo_first_hour.dropna()
h0 = exp[exp.y == 0].benzo_first_hour.dropna()
p = stats.mannwhitneyu(h1, h0).pvalue
print(f"\nAmong benzo-exposed: delirium first-dose hour median={h1.median():.1f} "
      f"vs no-delirium {h0.median():.1f}; Mann-Whitney p={p:.3f}")

out = pd.DataFrame(rows, columns=["Exposure definition", "Exposed n", "aOR", "95% CI"])
out.to_csv(os.path.join(ROOT, "output", "table_landmark.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(ROOT, "output", "table_landmark.md"), "w", encoding="utf-8") as f:
    f.write("# Landmark / early-window sensitivity: benzodiazepine & ABD (MIMIC-IV DEV)\n\n")
    f.write("Adjusted for: " + ", ".join(COV) + "\n\n")
    f.write(out.to_markdown(index=False))
    f.write(f"\n\nDirectionality: among benzo-exposed, delirium first-dose hour median "
            f"{h1.median():.1f}h vs no-delirium {h0.median():.1f}h (p={p:.3f}).\n")
