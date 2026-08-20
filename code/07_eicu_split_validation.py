# -*- coding: utf-8 -*-
"""
07_eicu_split_validation.py
用 eICU 内部-外部交叉验证（IECV, k=5 分层）替代"第 3 库独立验证"。

背景：NWICU（仅 32 例合格 SLE ICU、1 例事件、无 GCS）与 INSPIRE（非 ICU 库）
      均无法作为本研究独立验证集。故采用 TRIPOD 标准两库设计：
      MIMIC-IV 开发 → eICU 外部；并将 eICU 再拆分为
        (a) 再校准子集（model updating）与
        (b) 真正独立测试子集（IECV 中每个病例恰好作测试一次，再校准参数来自其他折）
      从而消除此前"再校准在同集拟合并评估"的乐观偏倚，且全程无 GCS（无循环论证）。

特征：与 04/05 一致，剔除 gcs_min_24h（35→34 候选）
结局：abd_dx_only（仅诊断编码，不含 GCS）—— 彻底无循环论证
输出：output/ 下 table11_iec_validation / fig_cal_iec_vs_naive / 元信息
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
# 1. 数据（复用 04/05 口径，剔除 GCS）
# ============================================================
mim = pd.read_csv(os.path.join(ROOT, "data", "mimic_sle_analytic.csv"))
eic = pd.read_csv(os.path.join(ROOT, "data", "eicu_sle_analytic.csv"))

dev = mim[(mim.cam_assessed == 1) & (mim.primary_brain_injury == 0)].copy().reset_index(drop=True)
dev["sofa_noncns"] = dev["sofa_24h"] - dev["sofa_cns"]
dev["y"] = dev["cam_pos"].astype(int)

# eICU 外部：排除原发脑损伤，不要求 GCS 记录（扩大集，与 06 一致）
ext = eic[(eic.primary_brain_injury == 0)].copy().reset_index(drop=True)
y_ext = ext["abd_dx_only"].astype(int).values

FEATS_ALL = [
    "age", "female", "sofa_noncns",
    "sepsis3_24h", "mech_vent_24h", "benzo_24h", "opioid_24h", "steroid_24h",
    "bicarbonate_min", "wbc_max", "platelets_min", "hemoglobin_min",
    "creatinine_max", "bun_max", "lactate_max", "glucose_max", "sodium_min",
    "temperature_max", "heart_rate_mean", "resp_rate_mean", "mbp_min",
    "propofol_24h", "dexmed_24h", "vaso_24h",
    "htn", "diabetes", "ckd", "esrd_dialysis", "epilepsy_hx",
    "alcohol_abuse", "chf", "malignancy", "lupus_nephritis_icd", "hcq",
]
FEATS = [f for f in FEATS_ALL if f in dev.columns and f in ext.columns]
LABEL = {
    "age": "年龄", "female": "女性", "sofa_noncns": "非神经 SOFA",
    "sepsis3_24h": "脓毒症", "mech_vent_24h": "有创通气", "benzo_24h": "苯二氮卓",
    "opioid_24h": "阿片类", "steroid_24h": "糖皮质激素", "bicarbonate_min": "碳酸氢根最低",
    "wbc_max": "白细胞最高", "platelets_min": "血小板最低", "hemoglobin_min": "血红蛋白最低",
    "creatinine_max": "肌酐最高", "bun_max": "尿素氮最高", "lactate_max": "乳酸最高",
    "glucose_max": "血糖最高", "sodium_min": "血钠最低", "temperature_max": "体温最高",
    "heart_rate_mean": "心率", "resp_rate_mean": "呼吸频率", "mbp_min": "平均动脉压最低",
    "propofol_24h": "丙泊酚", "dexmed_24h": "右美托咪定", "vaso_24h": "血管活性药",
    "htn": "高血压", "diabetes": "糖尿病", "ckd": "慢性肾病", "esrd_dialysis": "透析",
    "epilepsy_hx": "癫痫史", "alcohol_abuse": "酒精滥用", "chf": "心力衰竭",
    "malignancy": "恶性肿瘤", "lupus_nephritis_icd": "狼疮肾炎", "hcq": "羟氯喹",
}

print("=" * 70)
print(f"开发 (MIMIC-IV): N={len(dev)}  事件={dev.y.sum()} ({dev.y.mean()*100:.1f}%)")
print(f"外部 (eICU-CRD): N={len(ext)}  仅诊断结局事件={y_ext.sum()} ({y_ext.mean()*100:.1f}%)")
print(f"特征(已剔除GCS): {len(FEATS)} 个")
print("=" * 70)

Xd_raw, yd = dev[FEATS].astype(float), dev.y.values
Xe_raw = ext[FEATS].astype(float)
med = Xd_raw.median()
Xd, Xe = Xd_raw.fillna(med), Xe_raw.fillna(med)

# ============================================================
# 2. LASSO 选特征（开发队列，无 GCS）
# ============================================================
sc0 = StandardScaler().fit(Xd)
las = LogisticRegressionCV(Cs=30, cv=StratifiedKFold(10, shuffle=True, random_state=SEED),
                           penalty="l1", solver="saga", scoring="roc_auc",
                           max_iter=10000, random_state=SEED, n_jobs=-1).fit(sc0.transform(Xd), yd)
coef = pd.DataFrame({"feature": FEATS, "label": [LABEL.get(f, f) for f in FEATS], "coef": las.coef_[0]})
coef["abs"] = coef.coef.abs()
SEL = coef.loc[coef["abs"] > 1e-6, "feature"].tolist()
if len(SEL) < 5:
    SEL = coef.sort_values("abs", ascending=False).head(10).feature.tolist()
Xd_s, Xe_s = Xd[SEL], Xe[SEL]
print(f"[LASSO] C={las.C_[0]:.4f} 选中 {len(SEL)}/{len(FEATS)}: {[LABEL.get(f,f) for f in SEL]}")

# ============================================================
# 3. 模型定义
# ============================================================
cv10 = StratifiedKFold(10, shuffle=True, random_state=SEED)
MODELS = {
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
COLORS = {"Logistic 回归": "#185FA5", "LASSO-LR": "#0F6E56",
          "随机森林": "#993C1D", "XGBoost": "#534AB7"}

def auc_ci(y, p, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    a = roc_auc_score(y, p); bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        bs.append(roc_auc_score(y[idx], p[idx]))
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def perf(y, p):
    a, lo, hi = auc_ci(y, p)
    fpr, tpr, t = roc_curve(y, p); thr = t[np.argmax(tpr - fpr)]
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(AUC=a, AUC_lo=lo, AUC_hi=hi, Brier=brier_score_loss(y, p),
                Sens=tp/(tp+fn) if tp+fn else np.nan, Spec=tn/(tn+fp) if tn+fp else np.nan,
                PPV=tp/(tp+fp) if tp+fp else np.nan, NPV=tn/(tn+fn) if tn+fn else np.nan,
                Acc=(tp+tn)/len(y), thr=thr)

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

# ============================================================
# 4. 训练 MIMIC 原始模型 + 原始外部预测（无 eICU 拟合）
# ============================================================
fitted = {}
raw_pred = {}          # MIMIC 原始模型 → eICU（未再校准）
for nm, mdl in MODELS.items():
    mdl.fit(Xd_s, yd)
    fitted[nm] = mdl
    raw_pred[nm] = mdl.predict_proba(Xe_s)[:, 1]

# ============================================================
# 5. 再校准 + 内部-外部交叉验证（IECV）
#    eICU 5 折分层；每折在"训练折"拟再校准，预测"测试折"
#    朴素再校准：在全 256 拟合并评估（乐观基准）
# ============================================================
EPS = 1e-6
def fit_recal(y, p_raw):
    z = np.log(np.clip(p_raw, EPS, 1 - EPS) / (1 - np.clip(p_raw, EPS, 1 - EPS)))
    X = sm.add_constant(z)
    res = sm.Logit(y, X).fit(disp=0)
    return float(res.params[0]), float(res.params[1])   # a, b

def apply_recal(p_raw, a, b):
    z = np.log(np.clip(p_raw, EPS, 1 - EPS) / (1 - np.clip(p_raw, EPS, 1 - EPS)))
    return 1 / (1 + np.exp(-(a + b * z)))

K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
iec_pred = {nm: np.zeros(len(y_ext)) for nm in MODELS}
fold_info = {nm: [] for nm in MODELS}
for nm in MODELS:
    for tr, te in skf.split(Xe_s, y_ext):
        a, b = fit_recal(y_ext[tr], raw_pred[nm][tr])
        iec_pred[nm][te] = apply_recal(raw_pred[nm][te], a, b)
        fold_info[nm].append((a, b, len(te), int(y_ext[te].sum())))

# 朴素再校准（全 256 拟合并评估，乐观）
naive_pred = {}
for nm in MODELS:
    a, b = fit_recal(y_ext, raw_pred[nm])
    naive_pred[nm] = apply_recal(raw_pred[nm], a, b)

# ============================================================
# 6. 三种估计对比：原始 / 朴素再校准(乐观) / IECV(诚实)
# ============================================================
rows = []
for nm in MODELS:
    r_raw = perf(y_ext, raw_pred[nm])
    r_naive = perf(y_ext, naive_pred[nm])
    r_iec = perf(y_ext, iec_pred[nm])
    rows.append({
        "模型": nm,
        "原始 (无eICU拟合) AUC": f"{r_raw['AUC']:.3f} ({r_raw['AUC_lo']:.3f}-{r_raw['AUC_hi']:.3f})",
        "朴素再校准 AUC (乐观)": f"{r_naive['AUC']:.3f} ({r_naive['AUC_lo']:.3f}-{r_naive['AUC_hi']:.3f})",
        "IECV 诚实 AUC": f"{r_iec['AUC']:.3f} ({r_iec['AUC_lo']:.3f}-{r_iec['AUC_hi']:.3f})",
        "原始 Brier": round(r_raw["Brier"], 3),
        "朴素再校准 Brier": round(r_naive["Brier"], 3),
        "IECV Brier": round(r_iec["Brier"], 3),
        "朴素再校准 HL-P": round(hl_test(y_ext, naive_pred[nm])[1], 4),
        "IECV HL-P": round(hl_test(y_ext, iec_pred[nm])[1], 4),
        "_raw": r_raw["AUC"], "_naive": r_naive["AUC"], "_iec": r_iec["AUC"],
    })
    # 折级再校准参数（校准斜率/截距）
    a_mean = np.mean([fi[0] for fi in fold_info[nm]])
    b_mean = np.mean([fi[1] for fi in fold_info[nm]])
    print(f"  {nm:<12s} 原始AUC={r_raw['AUC']:.3f}  朴素AUC={r_naive['AUC']:.3f}  IECV_AUC={r_iec['AUC']:.3f}"
          f"  | IECV 校准 a={a_mean:.3f} b={b_mean:.3f}  HL-P={r_iec['AUC'] and hl_test(y_ext, iec_pred[nm])[1]:.3f}")

tab = pd.DataFrame(rows).drop(columns=["_raw", "_naive", "_iec"])
tab.to_csv(os.path.join(OUT, "table11_iec_validation.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "table11_iec_validation.md"), "w", encoding="utf-8") as f:
    f.write("# Table 11. eICU 内部-外部交叉验证（IECV, k=5）独立测试估计\n\n")
    f.write(f"- 开发队列：MIMIC-IV，N={len(dev)}，CAM-ICU 谵妄（无 GCS 特征）\n")
    f.write(f"- 外部队列：eICU-CRD，N={len(ext)}，结局=仅诊断编码（abd_dx_only，无 GCS），事件 {int(y_ext.sum())} 例（{y_ext.mean()*100:.1f}%）\n")
    f.write(f"- **IECV**：eICU 5 折分层，每折再校准(logit更新)在训练折拟合并预测测试折；每例恰作测试一次 → 无乐观偏倚、无 GCS 循环\n")
    f.write(f"- 对比：原始(无eICU拟合) / 朴素再校准(全256拟合并评估,乐观基准) / IECV(诚实)\n\n")
    f.write(tab.to_markdown(index=False))
    f.write("\n\n注：NWICU(32例/1事件/无GCS)与INSPIRE(非ICU库)均不可用，故以 eICU IECV 作为独立测试估计，符合 TRIPOD 两库内部-外部验证设计。\n")
print(f"\n[Table 11] 已保存。")

# ============================================================
# 7. 校准图：IECV(诚实) vs 朴素再校准(乐观) —— 以最佳模型为例
# ============================================================
best = tab.loc[tab["IECV Brier"].astype(float).idxmin(), "模型"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=160)
for ax, (key, preds, ttl) in zip(axes, [
        ("朴素再校准(乐观)", naive_pred, "朴素再校准（拟合并评估于同集，乐观）"),
        ("IECV(诚实)", iec_pred, "IECV 内部-外部验证（诚实估计）")]):
    for nm, p in preds.items():
        try:
            po, pp = calibration_curve(y_ext, p, n_bins=6, strategy="quantile")
            ax.plot(pp, po, "o-", color=COLORS[nm], lw=1.7, ms=4.5, label=nm)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], color="#B4B2A9", lw=0.9, ls="--", label="理想")
    ax.set_xlabel("预测概率", fontsize=10.5); ax.set_ylabel("实际发生率", fontsize=10.5)
    ax.set_title(f"{ttl}\n(结局=仅诊断, N={len(ext)}, 最佳={best})", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.suptitle("图 11. eICU 校准：朴素再校准(乐观) vs IECV 诚实估计", fontsize=12.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT, "fig_cal_iec_vs_naive.png"), bbox_inches="tight")
print("[图] fig_cal_iec_vs_naive.png")

# 全部 4 模型 ROC（IECV 诚实外部）
fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.6), dpi=160)
for nm, p in iec_pred.items():
    fpr, tpr, _ = roc_curve(y_ext, p)
    ax.plot(fpr, tpr, color=COLORS[nm], lw=1.9, label=f"{nm} AUC={roc_auc_score(y_ext, p):.3f}")
ax.plot([0, 1], [0, 1], color="#B4B2A9", lw=0.8, ls="--")
ax.set_xlabel("1 − 特异度", fontsize=10.5); ax.set_ylabel("敏感度", fontsize=10.5)
ax.set_title(f"IECV 诚实外部 ROC（eICU, N={len(ext)}）", fontsize=11)
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_roc_iec.png"), bbox_inches="tight")
print("[图] fig_roc_iec.png")

# ============================================================
# 8. 元信息
# ============================================================
meta = {
    "analysis": "eicu_internal_external_cv_validation",
    "rationale": "NWICU(32/1/no-GCS) and INSPIRE(non-ICU) infeasible; use eICU IECV as independent test",
    "seed": SEED, "k_folds": K,
    "dev": {"source": "MIMIC-IV", "n": int(len(dev)), "events": int(yd.sum()), "outcome": "CAM-ICU"},
    "ext": {"source": "eICU-CRD", "n": int(len(ext)), "events_dx_only": int(y_ext.sum())},
    "features_excluded": ["gcs_min_24h"],
    "selected_features": SEL, "lasso_C": float(las.C_[0]),
    "best_model_by_IECV_brier": best,
    "performance": tab.drop(columns=["原始 Brier","朴素再校准 Brier","IECV Brier",
                                     "朴素再校准 HL-P","IECV HL-P"]).to_dict(orient="records"),
}
with open(os.path.join(OUT, "iec_model_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
pd.DataFrame({f"raw_{k}": v for k, v in raw_pred.items()}).assign(y=y_ext).to_csv(
    os.path.join(OUT, "pred_iec_raw.csv"), index=False)
print("\n[完成] eICU IECV 独立验证结果已保存至 output/")
