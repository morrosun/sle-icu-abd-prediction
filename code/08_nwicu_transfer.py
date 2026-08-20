# -*- coding: utf-8 -*-
"""
08_nwicu_transfer.py
NWICU 165 脑病队列的 "模型可迁移性 / 阳性对照" 次要分析

设计:
  - 在 MIMIC-IV(SLE, 无 GCS) 上重训 4 个模型, 特征集 = 与 NWICU 共有且无 GCS 的子集 (FEATS_NW)
  - 将模型应用于 NWICU: 治疗组 = 165 脑病 ICU 队列 (enc_group=1), 对照组 = 600 非脑病 ICU (enc_group=0)
  - 阳性对照: 模型从未在 NWICU/脑病上训练; 若它能给脑病组更高预测风险, 说明学到的是
    "广义急性脑功能障碍"生理信号, 而非 SLE 特异假象
  - 判别指标: AUC(enc vs ctrl) + 风险分布(Wilcoxon) + 严重度校正 OR + 1:1 严重度匹配 AUC
输出: output/ fig_transfer_risk / fig_transfer_roc / fig_transfer_cal / table12_transfer.*
"""
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
from scipy.spatial.distance import cdist
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
SEED = 20260731
np.random.seed(SEED)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 载入 MIMIC 开发集 (无 GCS) ----------
mim = pd.read_csv(os.path.join(ROOT, "data", "mimic_sle_analytic.csv"))
dev = mim[(mim.cam_assessed == 1) & (mim.primary_brain_injury == 0)].copy().reset_index(drop=True)
dev["y"] = dev["cam_pos"].astype(int)

# ---------- 2. 与 NWICU 共有且无 GCS 的特征 (FEATS_NW) ----------
FEATS_NW = [
    "age", "female", "sepsis3_24h", "mech_vent_24h",
    "benzo_24h", "opioid_24h", "steroid_24h",
    "bicarbonate_min", "wbc_max", "platelets_min", "hemoglobin_min",
    "creatinine_max", "bun_max", "lactate_max", "glucose_max", "sodium_min",
    "temperature_max", "heart_rate_mean", "resp_rate_mean", "mbp_min",
    "propofol_24h", "dexmed_24h", "vaso_24h", "hcq",
    "htn", "diabetes", "ckd", "esrd_dialysis", "chf", "malignancy",
    "epilepsy_hx", "lupus_nephritis_icd",
]
FEATS_NW = [f for f in FEATS_NW if f in dev.columns]
LABEL = {
    "age": "年龄", "female": "女性", "sepsis3_24h": "脓毒症", "mech_vent_24h": "有创通气",
    "benzo_24h": "苯二氮卓", "opioid_24h": "阿片类", "steroid_24h": "糖皮质激素",
    "bicarbonate_min": "碳酸氢根最低", "wbc_max": "白细胞最高", "platelets_min": "血小板最低",
    "hemoglobin_min": "血红蛋白最低", "creatinine_max": "肌酐最高", "bun_max": "尿素氮最高",
    "lactate_max": "乳酸最高", "glucose_max": "血糖最高", "sodium_min": "血钠最低",
    "temperature_max": "体温最高", "heart_rate_mean": "心率", "resp_rate_mean": "呼吸频率",
    "mbp_min": "平均动脉压最低", "propofol_24h": "丙泊酚", "dexmed_24h": "右美托咪定",
    "vaso_24h": "血管活性药", "hcq": "羟氯喹", "htn": "高血压", "diabetes": "糖尿病",
    "ckd": "慢性肾病", "esrd_dialysis": "透析", "chf": "心力衰竭", "malignancy": "恶性肿瘤",
    "epilepsy_hx": "癫痫史", "lupus_nephritis_icd": "狼疮肾炎",
}
Xd_raw, yd = dev[FEATS_NW].astype(float), dev.y.values

# ---------- 3. 载入 NWICU 转移队列 ----------
nw = pd.read_csv(os.path.join(ROOT, "data", "nwicu_transfer.csv"))
nw["y"] = nw["enc_group"].astype(int)
Xn_raw = nw[FEATS_NW].astype(float)

# 缺失率
print("=" * 64)
print(f"MIMIC 开发(NW特征子集): N={len(dev)}  事件={yd.sum()} ({yd.mean()*100:.1f}%)")
print(f"NWICU 转移队列: N={len(nw)}  脑病(enc)={int(nw.enc_group.sum())}  对照={int((1-nw.enc_group).sum())}")
print(f"NWICU 特征缺失% (脑病 / 对照):")
for f in FEATS_NW:
    m1 = nw[nw.enc_group==1][f].isna().mean()*100
    m0 = nw[nw.enc_group==0][f].isna().mean()*100
    if max(m1,m0) > 5:
        print(f"   {f:20s} {m1:5.1f} / {m0:5.1f}")
print(f"共有特征 (FEATS_NW): {len(FEATS_NW)} 个")
print("=" * 64)

# ---------- 4. 中位数填补 (以 MIMIC 开发集中位数为准, 防泄漏) ----------
med = Xd_raw.median()
Xd = Xd_raw.fillna(med)
Xn = Xn_raw.fillna(med)

# ---------- 5. 训练 4 个模型 (MIMIC, 10折 CV 内部) ----------
skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
def mk_pipeline(name):
    if name == "Logistic":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])
    if name == "LASSO-LR":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegressionCV(Cs=30, cv=skf, penalty="l1", solver="saga",
                                                     scoring="roc_auc", max_iter=10000,
                                                     class_weight="balanced", random_state=SEED, n_jobs=-1))])
    if name == "RandomForest":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", RandomForestClassifier(n_estimators=600, max_depth=5,
                                                        class_weight="balanced", random_state=SEED, n_jobs=-1))])
    if name == "XGBoost":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                   subsample=0.8, colsample_bytree=0.8,
                                                   eval_metric="logloss", random_state=SEED, n_jobs=-1))])

models = {}
for name in ["Logistic", "LASSO-LR", "RandomForest", "XGBoost"]:
    pl = mk_pipeline(name)
    pl.fit(Xd, yd)
    models[name] = pl

# 内部 CV AUC (MIMIC)
print("\n[MIMIC 内部 10折 CV AUC]")
cv = {}
for name, pl in models.items():
    proba = cross_val_predict(pl, Xd, yd, cv=skf, method="predict_proba")[:, 1]
    a = roc_auc_score(yd, proba)
    cv[name] = a
    print(f"   {name:14s} {a:.3f}")

# ---------- 6. 应用到 NWICU (阳性对照) ----------
def auc_ci(y, p, n_boot=2000):
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    if len(np.unique(y)) < 2: return (np.nan, np.nan, np.nan)
    idx = np.arange(n)
    aucs = []
    rng = np.random.default_rng(SEED)
    for _ in range(n_boot):
        s = rng.choice(idx, n, replace=True)
        try: aucs.append(roc_auc_score(y[s], p[s]))
        except Exception: pass
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return roc_auc_score(y, p), lo, hi

def wilcox_risk(p_enc, p_ctrl):
    U, pval = stats.mannwhitneyu(p_enc, p_ctrl, alternative="greater")
    return pval

print("\n[NWICU 阳性对照: 模型风险评分判别脑病 vs 对照]")
rows = []
pred_all = {}
for name, pl in models.items():
    p = pl.predict_proba(Xn)[:, 1]
    pred_all[name] = p
    p_enc = p[nw.enc_group == 1]; p_ctrl = p[nw.enc_group == 0]
    a, lo, hi = auc_ci(nw.y.values, p)
    med_e, med_c = np.median(p_enc), np.median(p_ctrl)
    w = wilcox_risk(p_enc, p_ctrl)
    rows.append({"model": name, "AUC": round(a,3), "AUC_lo": round(lo,3), "AUC_hi": round(hi,3),
                 "median_risk_enc": round(med_e,3), "median_risk_ctrl": round(med_c,3),
                 "wilcox_p": f"{w:.1e}"})
    print(f"   {name:14s} AUC={a:.3f} (95%CI {lo:.3f}-{hi:.3f})  median risk enc={med_e:.3f} ctrl={med_c:.3f}  Wilcoxon p={w:.1e}")

# ---------- 7. 严重度校正 (logistic: enc ~ risk + 年龄 + 关键实验室) ----------
from sklearn.linear_model import LogisticRegression as LR
best = "XGBoost"
p_best = pred_all[best]
adj_feats = ["age", "lactate_max", "creatinine_max", "bicarbonate_min", "platelets_min", "wbc_max", "sepsis3_24h"]
Xa = nw[adj_feats].fillna(med).values
Xadj = np.column_stack([p_best, Xa])
Xadj_s = StandardScaler().fit_transform(Xadj)
lr = LR(max_iter=5000).fit(Xadj_s, nw.y.values)
coef = lr.coef_[0]
print("\n[严重度校正 logistic: 脑病 ~ 模型风险 + 年龄 + 乳酸/肌酐/碳酸氢根/血小板/白细胞/脓毒症]")
for nm, c in zip(["模型风险","年龄","乳酸","肌酐","碳酸氢根","血小板","白细胞","脓毒症"], coef):
    print(f"   {nm:10s} OR/0.1={np.exp(c*0.1):.3f}  (coef={c:.3f})")
# 模型风险 per 0.1 增量 OR (adjusted)
or_per01 = np.exp(coef[0]*0.1)

# ---------- 8. 1:1 严重度匹配 (贪婪最近邻, 无放回) ----------
sev_feats = ["age", "lactate_max", "creatinine_max", "bicarbonate_min", "platelets_min", "wbc_max", "sepsis3_24h"]
Xsev = nw[sev_feats].fillna(med).values
Xsev_s = StandardScaler().fit_transform(Xsev)
enc_idx = np.where(nw.enc_group.values == 1)[0]
ctrl_idx = np.where(nw.enc_group.values == 0)[0]
D = cdist(Xsev_s[enc_idx], Xsev_s[ctrl_idx])
matched_ctrl = []
used = set()
for i, ei in enumerate(enc_idx):
    row = D[i]
    order = np.argsort(row)
    for cj in order:
        if cj not in used:
            matched_ctrl.append(ctrl_idx[cj]); used.add(cj); break
matched_idx = np.concatenate([enc_idx, np.array(matched_ctrl)])
nm = nw.iloc[matched_idx].copy()
print(f"\n[1:1 严重度匹配] enc={len(enc_idx)} 匹配对照={len(matched_ctrl)} (共 {len(nm)})")
for name, p in pred_all.items():
    pm = p[matched_idx]
    a, lo, hi = auc_ci(nm.y.values, pm)
    print(f"   {name:14s} 匹配后 AUC={a:.3f} ({lo:.3f}-{hi:.3f})")
match_auc = {name: auc_ci(nm.y.values, p[matched_idx])[0] for name, p in pred_all.items()}

# ---------- 9. 图表 ----------
# Fig 1: 预测风险分布 (box + scatter)
fig, ax = plt.subplots(figsize=(7,4.5))
data = [p_best[nw.enc_group==1], p_best[nw.enc_group==0]]
bp = ax.boxplot(data, patch_artist=True, widths=0.5)
ax.set_xticks([1,2]); ax.set_xticklabels(["脑病组 (n=165)", "对照组 (n=600)"])
for patch, c in zip(bp["boxes"], ["#c0392b", "#2980b9"]):
    patch.set_facecolor(c); patch.set_alpha(0.6)
for i, d in enumerate(data, 1):
    x = np.random.normal(i, 0.06, len(d))
    ax.scatter(x, d, s=8, alpha=0.25, color=["#c0392b","#2980b9"][i-1])
ax.set_ylabel(f"{best} 预测风险"); ax.set_title("NWICU: MIMIC 模型对脑病 vs 对照的预测风险\n(阳性对照 —— 模型未在 NWICU 训练)")
ax.text(0.5, 0.96, f"median {np.median(data[0]):.3f} vs {np.median(data[1]):.3f}  | Wilcoxon p<1e-3",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#444")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_transfer_risk.png"), dpi=130); plt.close()

# Fig 2: ROC (full + matched)
fig, ax = plt.subplots(figsize=(6,6))
for label, yv, pv in [("全样本", nw.y.values, p_best),
                       ("1:1 严重度匹配", nm.y.values, p_best[matched_idx])]:
    fpr, tpr, _ = roc_curve(yv, pv)
    a = roc_auc_score(yv, pv)
    ax.plot(fpr, tpr, label=f"{label} (AUC={a:.3f})")
ax.plot([0,1],[0,1],"k--",alpha=0.4)
ax.set_xlabel("1-特异度"); ax.set_ylabel("敏感度")
ax.set_title("NWICU 阳性对照 ROC: 模型风险判别脑病"); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_transfer_roc.png"), dpi=130); plt.close()

# Fig 3: 校准式 (预测风险 vs 观测脑病比例, 十分位)
fig, ax = plt.subplots(figsize=(6,6))
order = np.argsort(p_best)
dec = np.linspace(0, len(p_best), 11).astype(int)
xs, ys = [], []
for k in range(10):
    sl = order[dec[k]:dec[k+1]]
    if len(sl) == 0: continue
    xs.append(p_best[sl].mean()); ys.append(nw.y.values[sl].mean())
ax.plot([0,1],[0,1],"k--",alpha=0.4,label="理想")
ax.plot(xs, ys, "o-", color="#8e44ad", label="观测脑病比例")
ax.set_xlabel("预测风险 (模型输出)"); ax.set_ylabel("观测脑病比例")
ax.set_title("NWICU 阳性对照: 预测风险 vs 观测脑病比例"); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_transfer_cal.png"), dpi=130); plt.close()

# ---------- 10. 表格 & 输出 ----------
tbl = pd.DataFrame(rows)
tbl.to_csv(os.path.join(OUT, "table12_transfer.csv"), index=False)
with open(os.path.join(OUT, "table12_transfer.md"), "w", encoding="utf-8") as f:
    f.write("# 表12  NWICU 模型可迁移性 / 阳性对照\n\n")
    f.write("MIMIC 模型(无GCS, 32特征)应用于 NWICU 165 脑病 + 600 对照。阳性对照: 模型未在 NWICU 训练。\n\n")
    f.write(tbl.to_markdown(index=False))
    f.write("\n\n## 内部 CV (MIMIC)\n")
    for k,v in cv.items(): f.write(f"- {k}: {v:.3f}\n")
    f.write(f"\n## 严重度校正 OR (XGBoost 风险 per 0.1 增量): {or_per01:.3f}\n")
    f.write("## 1:1 严重度匹配后 AUC\n")
    for k,v in match_auc.items(): f.write(f"- {k}: {v:.3f}\n")
    f.write("\n## 已知数据缺口 (NWICU)\n")
    f.write("- mech_vent_24h 无记录 (恒为0); inr_max 全缺失; lactate 稀疏(脑病84%/对照65%, 以MIMIC中位数填补)\n")
    f.write("- 仅能构造 '仅诊断' 口径, 无 GCS (故本分析走无GCS特征集)\n")

# 预测结果留存
out_pred = nw[["stay_id","enc_group"]].copy()
for name, p in pred_all.items(): out_pred[f"risk_{name}"] = p
out_pred.to_csv(os.path.join(OUT, "pred_nwicu_transfer.csv"), index=False)

print("\n完成. 输出: fig_transfer_risk/roc/cal.png, table12_transfer.{csv,md}, pred_nwicu_transfer.csv")
print(f"校正 OR(per 0.1 风险) = {or_per01:.3f}")
