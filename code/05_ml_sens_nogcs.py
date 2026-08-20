# -*- coding: utf-8 -*-
"""
05_ml_sens_nogcs.py
敏感性分析：剔除 GCS 预测因子，剥离"预测因子级"循环论证，给出干净外部 AUC。

敏感性 B（用户要求）：预测因子去掉 gcs_min_24h，外部结局仍用 abd_dx_gcs（诊断编码 OR GCS<=12）
敏感性 A（彻底无循环）：预测因子去掉 gcs_min_24h，外部结局改用 abd_dx_only（仅诊断编码，不含 GCS）

开发队列 : MIMIC-IV  N=264, 结局 CAM-ICU 谵妄
外部验证 : eICU-CRD  N=202, 排除原发脑损伤 + 有 GCS 记录
输出     : output/ 下 table7b(敏感性B) / table7c(仅诊断) / fig_roc_sens / fig_calibration_sens / fig_shap_sens*
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
# 1. 载入数据（与 04 完全一致，仅剔除 GCS 预测因子）
# ============================================================
mim = pd.read_csv(os.path.join(ROOT, "data", "mimic_sle_analytic.csv"))
eic = pd.read_csv(os.path.join(ROOT, "data", "eicu_sle_analytic.csv"))

dev = mim[(mim.cam_assessed == 1) & (mim.primary_brain_injury == 0)].copy().reset_index(drop=True)
dev["sofa_noncns"] = dev["sofa_24h"] - dev["sofa_cns"]
dev["y"] = dev["cam_pos"].astype(int)

val = eic[(eic.primary_brain_injury == 0) & (eic.gcs_min_24h.notna())].copy().reset_index(drop=True)
yv_B = val["abd_dx_gcs"].astype(int).values          # 敏感性 B 外部结局
yv_A = val["abd_dx_only"].astype(int).values         # 敏感性 A 外部结局（仅诊断，无 GCS）

# 两库共有候选变量（剔除 gcs_min_24h）
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
FEATS = [f for f in FEATS_ALL if f in dev.columns and f in val.columns]
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

print("=" * 66)
print(f"开发队列 (MIMIC-IV): N={len(dev):4d}  事件={dev.y.sum():3d} ({dev.y.mean()*100:.1f}%)")
print(f"验证队列 (eICU-CRD): N={len(val):4d}")
print(f"  外部结局 B (诊断+GCS): 事件={yv_B.sum():3d} ({yv_B.mean()*100:.1f}%)")
print(f"  外部结局 A (仅诊断)  : 事件={yv_A.sum():3d} ({yv_A.mean()*100:.1f}%)")
print(f"候选特征（已剔除 GCS）: {len(FEATS)} 个")
print("=" * 66)

Xd_raw, yd = dev[FEATS].astype(float), dev.y.values
Xv_raw = val[FEATS].astype(float)
med = Xd_raw.median()
Xd, Xv = Xd_raw.fillna(med), Xv_raw.fillna(med)

# ============================================================
# 2. LASSO 特征选择（10 折 CV，开发队列，无 GCS）
# ============================================================
sc0 = StandardScaler().fit(Xd)
las = LogisticRegressionCV(Cs=30, cv=StratifiedKFold(10, shuffle=True, random_state=SEED),
                           penalty="l1", solver="saga", scoring="roc_auc",
                           max_iter=10000, random_state=SEED, n_jobs=-1).fit(sc0.transform(Xd), yd)
coef = pd.DataFrame({"feature": FEATS, "label": [LABEL.get(f, f) for f in FEATS], "coef": las.coef_[0]})
coef["abs"] = coef.coef.abs()
SEL = coef.loc[coef["abs"] > 1e-6, "feature"].tolist()
coef.sort_values("abs", ascending=False).to_csv(
    os.path.join(OUT, "sens_lasso_coefficients_nogcs.csv"), index=False, encoding="utf-8-sig")
print(f"\n[LASSO 无GCS] 最优 C={las.C_[0]:.4f}，选中 {len(SEL)}/{len(FEATS)} 个特征")
print(coef[coef["abs"] > 1e-6].sort_values("abs", ascending=False)[["label", "coef"]].round(3).to_string(index=False))
if len(SEL) < 5:
    SEL = coef.sort_values("abs", ascending=False).head(10).feature.tolist()
    print("[提示] LASSO 过于稀疏，回退为绝对系数前 10 名")
Xd_s, Xv_s = Xd[SEL], Xv[SEL]

# ============================================================
# 3. 模型定义（与 04 一致）
# ============================================================
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
    rng = np.random.default_rng(seed)
    a = roc_auc_score(y, p); bs = []
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
                Sens=tp/(tp+fn) if tp+fn else np.nan, Spec=tn/(tn+fp) if tn+fp else np.nan,
                PPV=tp/(tp+fp) if tp+fp else np.nan, NPV=tn/(tn+fn) if tn+fn else np.nan,
                Acc=(tp+tn)/len(y), threshold=thr)


# ============================================================
# 4. 内部 10 折 CV + 双外部验证（B:诊断+GCS / A:仅诊断）
# ============================================================
res_rows, cv_preds, ext_preds_B, ext_preds_A, fitted = [], {}, {}, {}, {}
for nm, mdl in models.items():
    pcv = cross_val_predict(mdl, Xd_s, yd, cv=cv10, method="predict_proba", n_jobs=-1)[:, 1]
    cv_preds[nm] = pcv
    m = mdl.fit(Xd_s, yd); fitted[nm] = m
    pex_B = m.predict_proba(Xv_s)[:, 1]; ext_preds_B[nm] = pex_B
    pex_A = pex_B                              # 同一模型对同一批患者，换结局评估
    ext_preds_A[nm] = pex_A

    r_cv = perf(yd, pcv)
    r_B = perf(yv_B, pex_B)
    r_A = perf(yv_A, pex_A)
    res_rows.append({
        "模型": nm,
        "内部 CV AUC (95% CI)": f"{r_cv['AUC']:.3f} ({r_cv['AUC_lo']:.3f}-{r_cv['AUC_hi']:.3f})",
        "外部B AUC (诊断+GCS)": f"{r_B['AUC']:.3f} ({r_B['AUC_lo']:.3f}-{r_B['AUC_hi']:.3f})",
        "外部A AUC (仅诊断)": f"{r_A['AUC']:.3f} ({r_A['AUC_lo']:.3f}-{r_A['AUC_hi']:.3f})",
        "外部A Brier": f"{r_A['Brier']:.3f}",
        "外部A 敏感度": f"{r_A['Sens']:.3f}", "外部A 特异度": f"{r_A['Spec']:.3f}",
        "外部A PPV": f"{r_A['PPV']:.3f}", "外部A NPV": f"{r_A['NPV']:.3f}",
        "_auc_cv": r_cv["AUC"], "_auc_B": r_B["AUC"], "_auc_A": r_A["AUC"],
    })
    print(f"  {nm:<14s} 内部CV={r_cv['AUC']:.3f}  外部B(诊断+GCS)={r_B['AUC']:.3f}  外部A(仅诊断)={r_A['AUC']:.3f}")

tab = pd.DataFrame(res_rows)
bestA = tab.loc[tab._auc_A.idxmax(), "模型"]
tab_out = tab.drop(columns=["_auc_cv", "_auc_B", "_auc_A"])
tab_out.to_csv(os.path.join(OUT, "table7c_sens_nogcs.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "table7c_sens_nogcs.md"), "w", encoding="utf-8") as f:
    f.write("# Table 7C. 敏感性分析（剔除 GCS 预测因子）模型性能\n\n")
    f.write(f"- 开发队列：MIMIC-IV，N={len(dev)}，CAM-ICU 谵妄 {yd.sum()} 例（{yd.mean()*100:.1f}%）\n")
    f.write(f"- 外部验证：eICU-CRD，N={len(val)}，预测因子**已剔除 gcs_min_24h**\n")
    f.write(f"- 外部B 结局：诊断编码 OR GCS≤12（{yv_B.sum()} 例，{yv_B.mean()*100:.1f}%）\n")
    f.write(f"- 外部A 结局：**仅诊断编码**（{yv_A.sum()} 例，{yv_A.mean()*100:.1f}%）—— 完全无 GCS 循环论证\n")
    f.write(f"- 特征：LASSO 从 {len(FEATS)} 个候选中筛选 {len(SEL)} 个\n\n")
    f.write(tab_out.to_markdown(index=False))
    f.write(f"\n\n外部A（仅诊断）最佳模型：**{bestA}**\n")
    f.write("\n注：外部B 仍存在'结局含 GCS≤12'的残留重叠，外部A 两端均无 GCS，是真正的无循环外部 AUC。\n")
print(f"\n[Table 7C] 已保存。外部A(仅诊断)最佳：{bestA}")

# ============================================================
# 5. ROC（双面板：外部B / 外部A）
# ============================================================
COLORS = {"Logistic 回归": "#185FA5", "LASSO-LR": "#0F6E56",
          "随机森林": "#993C1D", "XGBoost": "#534AB7"}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=160)
for ax, (preds, yy, ttl) in zip(axes, [
        (ext_preds_B, yv_B, f"外部 B：诊断+GCS（eICU, N={len(val)}）"),
        (ext_preds_A, yv_A, f"外部 A：仅诊断（eICU, N={len(val)}）")]):
    for nm, p in preds.items():
        fpr, tpr, _ = roc_curve(yy, p)
        ax.plot(fpr, tpr, color=COLORS[nm], lw=1.9, label=f"{nm}  AUC={roc_auc_score(yy, p):.3f}")
    ax.plot([0, 1], [0, 1], color="#B4B2A9", lw=0.8, ls="--")
    ax.set_xlabel("1 − 特异度", fontsize=10.5); ax.set_ylabel("敏感度", fontsize=10.5)
    ax.set_title(ttl, fontsize=11); ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
fig.suptitle("图 S1. 敏感性验证 ROC（剔除 GCS 预测因子）", fontsize=12.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT, "fig_roc_sens.png"), bbox_inches="tight")
print("[图] fig_roc_sens.png")

# ============================================================
# 6. 校准（外部A：仅诊断，最干净）
# ============================================================
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

fig, ax = plt.subplots(1, 1, figsize=(5.8, 5.2), dpi=160)
cal_rows = []
for nm, p in ext_preds_A.items():
    try:
        po, pp = calibration_curve(yv_A, p, n_bins=6, strategy="quantile")
        ax.plot(pp, po, "o-", color=COLORS[nm], lw=1.7, ms=4.5, label=nm)
    except Exception:
        pass
    chi2, ph = hl_test(yv_A, p)
    cal_rows.append(dict(模型=nm, HL_chi2=round(chi2, 3), HL_P=round(ph, 4),
                         Brier=round(brier_score_loss(yv_A, p), 3)))
ax.plot([0, 1], [0, 1], color="#B4B2A9", lw=0.9, ls="--", label="理想校准")
ax.set_xlabel("预测概率", fontsize=10.5); ax.set_ylabel("实际发生率", fontsize=10.5)
ax.set_title(f"外部 A 校准（仅诊断, N={len(val)}）", fontsize=11)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.suptitle("图 S2. 外部 A（仅诊断）校准曲线", fontsize=12.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT, "fig_calibration_sens.png"), bbox_inches="tight")
pd.DataFrame(cal_rows).to_csv(os.path.join(OUT, "table8c_sens_calibration.csv"), index=False, encoding="utf-8-sig")
print("[图] fig_calibration_sens.png  [表] table8c_sens_calibration.csv")

# ============================================================
# 7. SHAP（敏感性 B/XGBoost，无 GCS）
# ============================================================
try:
    import shap
    xgbm = fitted["XGBoost"].named_steps["clf"]
    Xs = Xd_s.copy(); Xs.columns = [LABEL.get(c, c) for c in Xs.columns]
    expl = shap.TreeExplainer(xgbm); sv = expl.shap_values(Xs)
    if isinstance(sv, (list, tuple)):
        sv = sv[1] if len(sv) > 1 else sv[0]
    assert sv.ndim == 2

    plt.figure(figsize=(8.2, 6.2), dpi=160)
    shap.summary_plot(sv, Xs, show=False, max_display=15, plot_size=None)
    plt.title("图 S3A. SHAP 重要性（剔除 GCS, XGBoost）", fontsize=11.5)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_shap_sens_beeswarm.png"), bbox_inches="tight"); plt.close()

    plt.figure(figsize=(7.6, 5.6), dpi=160)
    shap.summary_plot(sv, Xs, plot_type="bar", show=False, max_display=15, plot_size=None)
    plt.title("图 S3B. SHAP 平均绝对贡献度（剔除 GCS）", fontsize=11.5)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_shap_sens_bar.png"), bbox_inches="tight"); plt.close()

    imp = pd.DataFrame({"特征": Xs.columns, "平均|SHAP|": np.abs(sv).mean(0)}).sort_values("平均|SHAP|", ascending=False)
    imp.round(4).to_csv(os.path.join(OUT, "shap_importance_nogcs.csv"), index=False, encoding="utf-8-sig")
    print("[图] fig_shap_sens_beeswarm.png / fig_shap_sens_bar.png")
    print("\n[SHAP 重要性 Top 10（无 GCS）]")
    print(imp.head(10).round(4).to_string(index=False))
except ImportError:
    print("[跳过 SHAP] shap 未安装")

# ============================================================
# 8. 元信息与预测概率
# ============================================================
meta = {
    "analysis": "sensitivity_exclude_gcs_predictor",
    "seed": SEED,
    "dev_cohort": {"source": "MIMIC-IV", "n": int(len(dev)), "events": int(yd.sum()), "outcome": "CAM-ICU"},
    "val_cohort": {"source": "eICU-CRD", "n": int(len(val)),
                   "events_B_dx_gcs": int(yv_B.sum()), "events_A_dx_only": int(yv_A.sum())},
    "features_excluded": ["gcs_min_24h"],
    "candidate_features": FEATS, "selected_features": SEL, "lasso_C": float(las.C_[0]),
    "best_externalA_model": bestA,
    "performance": tab_out.to_dict(orient="records"),
}
with open(os.path.join(OUT, "sens_model_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
pd.DataFrame({**{f"valB_{k}": v for k, v in ext_preds_B.items()},
              **{f"valA_{k}": v for k, v in ext_preds_A.items()},
              "val_y_B": yv_B, "val_y_A": yv_A}).to_csv(os.path.join(OUT, "pred_sens_external.csv"), index=False)
print("\n[完成] 敏感性分析全部结果已保存至 output/")
