# -*- coding: utf-8 -*-
"""
06_recal_expand.py
夯实跨库结论的两项补充分析：
  (A) eICU 截距/斜率再校准（TRIPOD 模型更新）：对 MIMIC 训练的模型在 eICU 上做 logit 再校准
      logit(p_cal) = a + b * logit(p_raw)
      分别在两个外部结局上做：abd_dx_gcs（需GCS子集202）、abd_dx_only（扩大256）
  (B) 扩大"仅诊断"外部样本：不要求 GCS 记录，eICU 排除脑损伤后全部 256 例，用 abd_dx_only 评估

输出 output/:
  table9_recalibration.md/csv  再校准前后指标（主外部 abd_dx_gcs）
  table10_expanded_dxonly.md/csv 扩大样本干净外部 AUC（abd_dx_only, 256）
  fig_cal_recal.png            再校准前后校准曲线（两结局）
"""
import os, warnings, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             average_precision_score, confusion_matrix)
from sklearn.calibration import calibration_curve
from scipy import stats
import statsmodels.api as sm
import xgboost as xgb

warnings.filterwarnings("ignore")
SEED = 20260731
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 数据（与 05 一致：剔除 GCS 预测因子）
# ============================================================
mim = pd.read_csv(os.path.join(ROOT, "data", "mimic_sle_analytic.csv"))
eic = pd.read_csv(os.path.join(ROOT, "data", "eicu_sle_analytic.csv"))

dev = mim[(mim.cam_assessed == 1) & (mim.primary_brain_injury == 0)].copy().reset_index(drop=True)
dev["sofa_noncns"] = dev["sofa_24h"] - dev["sofa_cns"]
dev["y"] = dev["cam_pos"].astype(int)

val_gcs = eic[(eic.primary_brain_injury == 0) & (eic.gcs_min_24h.notna())].copy().reset_index(drop=True)  # 202
val_exp = eic[(eic.primary_brain_injury == 0)].copy().reset_index(drop=True)                              # 256 扩大

y_B = val_gcs["abd_dx_gcs"].astype(int).values          # 主外部（需GCS子集）
y_A = val_exp["abd_dx_only"].astype(int).values         # 扩大干净外部（256）

FEATS = [
    "age", "female", "sofa_noncns",
    "sepsis3_24h", "mech_vent_24h", "benzo_24h", "opioid_24h", "steroid_24h",
    "bicarbonate_min", "wbc_max", "platelets_min", "hemoglobin_min",
    "creatinine_max", "bun_max", "lactate_max", "glucose_max", "sodium_min",
    "temperature_max", "heart_rate_mean", "resp_rate_mean", "mbp_min",
    "propofol_24h", "dexmed_24h", "vaso_24h",
    "htn", "diabetes", "ckd", "esrd_dialysis", "epilepsy_hx",
    "alcohol_abuse", "chf", "malignancy", "lupus_nephritis_icd", "hcq",
]
FEATS = [f for f in FEATS if f in dev.columns and f in val_exp.columns]

Xd_raw, yd = dev[FEATS].astype(float), dev.y.values
XvB_raw = val_gcs[FEATS].astype(float)
XvA_raw = val_exp[FEATS].astype(float)
med = Xd_raw.median()
Xd, XvB, XvA = Xd_raw.fillna(med), XvB_raw.fillna(med), XvA_raw.fillna(med)

print("=" * 66)
print(f"开发 (MIMIC-IV): N={len(dev)} 事件={yd.sum()} ({yd.mean()*100:.1f}%)")
print(f"外部B 子集(需GCS): N={len(val_gcs)} 事件(abd_dx_gcs)={y_B.sum()} ({y_B.mean()*100:.1f}%)")
print(f"外部A 扩大(无GCS要求): N={len(val_exp)} 事件(abd_dx_only)={y_A.sum()} ({y_A.mean()*100:.1f}%)")
print(f"候选特征（无GCS）: {len(FEATS)}")
print("=" * 66)

# ============================================================
# 2. LASSO 选特征 + 模型（与 05 同口径）
# ============================================================
sc0 = StandardScaler().fit(Xd)
las = LogisticRegressionCV(Cs=30, cv=StratifiedKFold(10, shuffle=True, random_state=SEED),
                           penalty="l1", solver="saga", scoring="roc_auc",
                           max_iter=10000, random_state=SEED, n_jobs=-1).fit(sc0.transform(Xd), yd)
coef = pd.DataFrame({"feature": FEATS, "coef": las.coef_[0]}); coef["abs"] = coef.coef.abs()
SEL = coef.loc[coef["abs"] > 1e-6, "feature"].tolist()
if len(SEL) < 5:
    SEL = coef.sort_values("abs", ascending=False).head(10).feature.tolist()
Xd_s, XvB_s, XvA_s = Xd[SEL], XvB[SEL], XvA[SEL]
print(f"[LASSO] C={las.C_[0]:.4f} 选中 {len(SEL)} 个特征")

cv10 = StratifiedKFold(10, shuffle=True, random_state=SEED)
models = {
    "Logistic 回归": Pipeline([("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
    "LASSO-LR": Pipeline([("sc", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="saga", C=las.C_[0],
                                   max_iter=10000, random_state=SEED))]),
    "随机森林": Pipeline([("clf", RandomForestClassifier(n_estimators=600, max_depth=5,
            min_samples_leaf=8, max_features="sqrt", class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1))]),
    "XGBoost": Pipeline([("clf", xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=5,
            eval_metric="logloss", random_state=SEED, n_jobs=-1))]),
}


def auc_ci(y, p, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed); a = roc_auc_score(y, p); bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        bs.append(roc_auc_score(y[idx], p[idx]))
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def perf(y, p, thr=None):
    a, lo, hi = auc_ci(y, p)
    if thr is None:
        fpr, tpr, t = roc_curve(y, p); thr = t[np.argmax(tpr - fpr)]
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(AUC=a, AUC_lo=lo, AUC_hi=hi, Brier=brier_score_loss(y, p),
                Sens=tp/(tp+fn) if tp+fn else np.nan, Spec=tn/(tn+fp) if tn+fp else np.nan)


def hl_test(y, p, g=10):
    d = pd.DataFrame({"y": y, "p": p})
    try:
        d["grp"] = pd.qcut(d.p, g, labels=False, duplicates="drop")
    except ValueError:
        return np.nan, np.nan
    gg = d.groupby("grp").agg(obs=("y", "sum"), n=("y", "size"), exp=("p", "sum"))
    gg = gg[(gg.exp > 0) & (gg.n - gg.exp > 0)]
    chi2 = (((gg.obs - gg.exp) ** 2) / (gg.exp * (1 - gg.exp / gg.n))).sum()
    return float(chi2), float(stats.chi2.sf(chi2, max(1, len(gg) - 2)))


def recalibrate(y, p):
    """截距+斜率再校准：logit(p_cal)=a+b*logit(p_raw)"""
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    res = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    a, b = res.params[0], res.params[1]
    return 1 / (1 + np.exp(-(a + b * lp))), a, b


def recal_intercept_only(y, p):
    """仅截距再校准（斜率固定=1）：logit(p_cal)=a+logit(p_raw)"""
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    res = sm.Logit(y, lp).fit(disp=0)
    a = res.params[0]
    return 1 / (1 + np.exp(-(a + lp))), a, 1.0


# ============================================================
# 3. (A) 主外部再校准（abd_dx_gcs，202子集）
# ============================================================
print("\n[A] 主外部再校准（abd_dx_gcs, N=%d）" % len(val_gcs))
rows9 = []
raw_B, cal_B, calB_int = {}, {}, {}
for nm, mdl in models.items():
    m = mdl.fit(Xd_s, yd)
    p_raw = m.predict_proba(XvB_s)[:, 1]
    p_cal, a, b = recalibrate(y_B, p_raw)
    p_ci, ai, bi = recal_intercept_only(y_B, p_raw)
    raw_B[nm], cal_B[nm], calB_int[nm] = p_raw, p_cal, p_ci

    r_raw = perf(y_B, p_raw); r_cal = perf(y_B, p_cal); r_ci = perf(y_B, p_ci)
    chi_raw, ph_raw = hl_test(y_B, p_raw)
    chi_cal, ph_cal = hl_test(y_B, p_cal)
    rows9.append({
        "模型": nm,
        "原始 AUC": f"{r_raw['AUC']:.3f}", "原始 Brier": f"{r_raw['Brier']:.3f}",
        "原始 HL-P": f"{ph_raw:.3f}",
        "再校准 AUC": f"{r_cal['AUC']:.3f}", "再校准 Brier": f"{r_cal['Brier']:.3f}",
        "再校准 HL-P": f"{ph_cal:.3f}", "斜率b": f"{b:.3f}", "截距a": f"{a:.3f}",
        "仅截距HL-P": f"{hl_test(y_B, p_ci)[1]:.3f}",
    })
    print(f"  {nm:<14s} AUC {r_raw['AUC']:.3f}→{r_cal['AUC']:.3f} (不变)  "
          f"Brier {r_raw['Brier']:.3f}→{r_cal['Brier']:.3f}  HL-P {ph_raw:.3f}→{ph_cal:.3f}  b={b:.3f} a={a:.3f}")
tab9 = pd.DataFrame(rows9)
tab9.to_csv(os.path.join(OUT, "table9_recalibration.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "table9_recalibration.md"), "w", encoding="utf-8") as f:
    f.write("# Table 9. eICU 主外部再校准（abd_dx_gcs, N=202）\n\n")
    f.write("- 模型在 MIMIC 训练（剔除 GCS 预测因子），在 eICU 做 logit 截距+斜率再校准\n")
    f.write("- AUC 为单调变换，再校准前后不变；Brier / HL-P 改善说明校准度提升\n\n")
    f.write(tab9.to_markdown(index=False))
print("[表] table9_recalibration.md")

# ============================================================
# 4. (B) 扩大干净外部（abd_dx_only，256）+ 再校准
# ============================================================
print("\n[B] 扩大干净外部（abd_dx_only, N=%d）" % len(val_exp))
rows10 = []
raw_A, cal_A = {}, {}
bestA_auc = -1; bestA_nm = ""
for nm, mdl in models.items():
    m = mdl.fit(Xd_s, yd)
    p_raw = m.predict_proba(XvA_s)[:, 1]
    p_cal, a, b = recalibrate(y_A, p_raw)
    raw_A[nm], cal_A[nm] = p_raw, p_cal

    r_raw = perf(y_A, p_raw); r_cal = perf(y_A, p_cal)
    chi_raw, ph_raw = hl_test(y_A, p_raw)
    chi_cal, ph_cal = hl_test(y_A, p_cal)
    rows10.append({
        "模型": nm,
        "原始 AUC (95% CI)": f"{r_raw['AUC']:.3f} ({r_raw['AUC_lo']:.3f}-{r_raw['AUC_hi']:.3f})",
        "原始 Brier": f"{r_raw['Brier']:.3f}", "原始 HL-P": f"{ph_raw:.3f}",
        "再校准 AUC": f"{r_cal['AUC']:.3f}", "再校准 Brier": f"{r_cal['Brier']:.3f}",
        "再校准 HL-P": f"{ph_cal:.3f}", "斜率b": f"{b:.3f}", "截距a": f"{a:.3f}",
        "敏感度": f"{r_cal['Sens']:.3f}", "特异度": f"{r_cal['Spec']:.3f}",
    })
    if r_raw["AUC"] > bestA_auc:
        bestA_auc, bestA_nm = r_raw["AUC"], nm
    print(f"  {nm:<14s} 扩大外部AUC {r_raw['AUC']:.3f} ({r_raw['AUC_lo']:.3f}-{r_raw['AUC_hi']:.3f})  "
          f"Brier {r_raw['Brier']:.3f}→{r_cal['Brier']:.3f}  HL-P {ph_raw:.3f}→{ph_cal:.3f}")
tab10 = pd.DataFrame(rows10)
tab10.to_csv(os.path.join(OUT, "table10_expanded_dxonly.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "table10_expanded_dxonly.md"), "w", encoding="utf-8") as f:
    f.write("# Table 10. 扩大样本干净外部验证（abd_dx_only, N=256, 无 GCS 要求）\n\n")
    f.write(f"- 外部集扩大至 eICU 排除脑损伤后全部 {len(val_exp)} 例（不再要求 GCS 记录）\n")
    f.write(f"- 结局仅诊断编码（abd_dx_only），完全无 GCS 循环论证，事件 {y_A.sum()} 例（{y_A.mean()*100:.1f}%）\n")
    f.write(f"- 模型同 05（剔除 GCS 预测因子）\n\n")
    f.write(tab10.to_markdown(index=False))
    f.write(f"\n\n扩大样本最佳外部 AUC 模型：**{bestA_nm}**\n")
print(f"[表] table10_expanded_dxonly.md  最佳：{bestA_nm}")

# ============================================================
# 5. 再校准前后校准曲线图（XGBoost 为代表）
# ============================================================
COLORS = {"Logistic 回归": "#185FA5", "LASSO-LR": "#0F6E56",
          "随机森林": "#993C1D", "XGBoost": "#534AB7"}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=160)
# 左：主外部 abd_dx_gcs（202）
ax = axes[0]
ax.plot([0, 1], [0, 1], color="#B4B2A9", lw=0.9, ls="--", label="理想")
for nm in models:
    try:
        po, pp = calibration_curve(y_B, cal_B[nm], n_bins=6, strategy="quantile")
        ax.plot(pp, po, "o-", color=COLORS[nm], lw=1.6, ms=4, label=f"{nm} 校准后")
    except Exception:
        pass
ax.set_title(f"主外部 abd_dx_gcs（N={len(val_gcs)}）再校准后", fontsize=10.5)
ax.set_xlabel("预测概率"); ax.set_ylabel("实际发生率")
ax.legend(loc="upper left", fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
# 右：扩大干净外部 abd_dx_only（256）
ax = axes[1]
ax.plot([0, 1], [0, 1], color="#B4B2A9", lw=0.9, ls="--", label="理想")
for nm in models:
    try:
        po, pp = calibration_curve(y_A, cal_A[nm], n_bins=6, strategy="quantile")
        ax.plot(pp, po, "o-", color=COLORS[nm], lw=1.6, ms=4, label=f"{nm} 校准后")
    except Exception:
        pass
ax.set_title(f"扩大干净外部 abd_dx_only（N={len(val_exp)}）再校准后", fontsize=10.5)
ax.set_xlabel("预测概率")
ax.legend(loc="upper left", fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.suptitle("图 S4. eICU 再校准后校准曲线", fontsize=12.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT, "fig_cal_recal.png"), bbox_inches="tight")
print("[图] fig_cal_recal.png")

# ============================================================
# 6. 元信息
# ============================================================
meta = {
    "analysis": "eicu_recalibration_and_expanded_dxonly",
    "recalibration": "logit(p_cal)=a+b*logit(p_raw) on eICU outcomes",
    "main_external": {"outcome": "abd_dx_gcs", "n": int(len(val_gcs)), "events": int(y_B.sum())},
    "expanded_clean": {"outcome": "abd_dx_only", "n": int(len(val_exp)), "events": int(y_A.sum()),
                       "note": "no GCS requirement, no circularity"},
    "best_expanded_model": bestA_nm,
}
with open(os.path.join(OUT, "recal_expand_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("\n[完成] 再校准 + 扩大样本分析已保存至 output/")
