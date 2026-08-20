# -*- coding: utf-8 -*-
"""SLECI: internal-validation optimism (apparent vs 10-fold CV AUC) for DEV.
Supports the EPV / sample-size limitation discussion (Riley 2019, van Smeden 2019)."""
import os, numpy as np, pandas as pd, warnings
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = pd.read_csv(os.path.join(ROOT, "data", "mimic_sle_analytic.csv"))
dev = m[(m.cam_assessed == 1) & (m.primary_brain_injury == 0)].copy().reset_index(drop=True)
dev["sofa_noncns"] = dev["sofa_24h"] - dev["sofa_cns"]
dev["y"] = dev["cam_pos"].astype(int)

FEATS = ["age","female","sofa_noncns","gcs_min_24h","sepsis3_24h","mech_vent_24h",
         "benzo_24h","opioid_24h","steroid_24h","bicarbonate_min","wbc_max","platelets_min",
         "hemoglobin_min","creatinine_max","bun_max","lactate_max","glucose_max","sodium_min",
         "temperature_max","heart_rate_mean","resp_rate_mean","mbp_min","propofol_24h",
         "dexmed_24h","vaso_24h","htn","diabetes","ckd","esrd_dialysis","epilepsy_hx",
         "alcohol_abuse","chf","malignancy","lupus_nephritis_icd","hcq"]
FEATS = [f for f in FEATS if f in dev.columns]
Xd = dev[FEATS].astype(float); yd = dev.y.values
med = Xd.median(); Xd = Xd.fillna(med)
print(f"DEV N={len(dev)} events={int(yd.sum())} candidate predictors={len(FEATS)}  EPV={yd.sum()/len(FEATS):.2f}")

sc = StandardScaler().fit(Xd)
models = {
    "Logistic": Pipeline([("imp",SimpleImputer(strategy="median")),
                          ("clf",LogisticRegression(max_iter=2000,class_weight="balanced"))]),
    "LASSO-LR": Pipeline([("imp",SimpleImputer(strategy="median")),
                          ("sc",StandardScaler()),
                          ("clf",LogisticRegressionCV(Cs=30,cv=StratifiedKFold(10,shuffle=True,random_state=1),
                              penalty="l1",solver="saga",scoring="roc_auc",max_iter=10000))]),
    "RandomForest": RandomForestClassifier(n_estimators=600,max_depth=5,random_state=1,class_weight="balanced",n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,
                                 eval_metric="logloss",random_state=1,use_label_encoder=False),
}
skf = StratifiedKFold(10, shuffle=True, random_state=1)
rows = []
for name, clf in models.items():
    if name == "Logistic" or name == "LASSO-LR":
        Xt = sc.transform(Xd) if name=="Logistic" else Xd
        # apparent AUC on training (resubstitution)
        clf.fit(Xt, yd); p_app = clf.predict_proba(Xt)[:,1]
    else:
        clf.fit(Xd, yd); p_app = clf.predict_proba(Xd)[:,1]
    app = roc_auc_score(yd, p_app)
    p_cv = cross_val_predict(clf, sc.transform(Xd) if name=="Logistic" else Xd, yd, cv=skf, method="predict_proba")[:,1]
    cv = roc_auc_score(yd, p_cv)
    rows.append([name, f"{app:.3f}", f"{cv:.3f}", f"{app-cv:.3f}"])
    print(f"{name:14s} apparent={app:.3f}  CV={cv:.3f}  optimism={app-cv:.3f}")

t = pd.DataFrame(rows, columns=["Model","Apparent AUC","10-fold CV AUC","Optimism (app-CV)"])
t.to_csv(os.path.join(ROOT,"output","table_optimism.csv"),index=False,encoding="utf-8-sig")
with open(os.path.join(ROOT,"output","table_optimism.md"),"w",encoding="utf-8") as f:
    f.write("# Internal-validation optimism (MIMIC-IV DEV, N=264, events=105)\n\n")
    f.write(f"Candidate predictors={len(FEATS)}; EPV={yd.sum()/len(FEATS):.2f} (Riley 2019 suggests EPV>=10-20 for stable estimation).\n\n")
    f.write(t.to_markdown(index=False))
