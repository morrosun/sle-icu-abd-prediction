# -*- coding: utf-8 -*-
"""
04_ml_model.py
SLE 重症患者急性脑功能障碍预测模型（TRIPOD+AI）

开发队列 : MIMIC-IV  N=264, 结局 CAM-ICU 谵妄
验证队列 : eICU-CRD  N=202, 结局 诊断编码 + GCS<=12

流程:
  1. 特征对齐（两库共有的入 ICU 24h 变量）
  2. LASSO 特征选择（10 折 CV）
  3. 四个模型: Logistic / LASSO-LR / RandomForest / XGBoost
  4. 10 折分层交叉验证（内部）+ eICU 外部验证
  5. ROC / 校准曲线 / DCA / SHAP
输出: output/ 下 table7、fig_roc、fig_calibration、fig_dca、fig_shap*
"""
import os, warnings, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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
# 1. 载入数据 + 特征对齐
# ============================================================
mim = pd.read_csv(os.path.join(ROOT, "data", "mimic_sle_analytic.csv"))
eic = pd.read_csv(os.path.join(ROOT, "data", "eicu_sle_analytic.csv"))

# 开发集：MIMIC，CAM 已评估 + 排除原发脑损伤
dev = mim[(mim.cam_assessed == 1) & (mim.primary_brain_injury == 0)].copy().reset_index(drop=True)
dev["sofa_noncns"] = dev["sofa_24h"] - dev["sofa_cns"]
dev["y"] = dev["cam_pos"].astype(int)

# 验证集：eICU，排除原发脑损伤 + 有 GCS 记录（结局定义依赖 GCS）
val = eic[(eic.primary_brain_injury == 0) & (eic.gcs_min_24h.notna())].copy().reset_index(drop=True)
val["y"] = val["abd_dx_gcs"].astype(int)

# 两库共有的入 ICU 24h 预测变量
FEATS = [
    "age", "female", "sofa_noncns", "gcs_min_24h",
    "sepsis3_24h", "mech_vent_24h", "benzo_24h", "opioid_24h", "steroid_24h",
    "bicarbonate_min", "wbc_max", "platelets_min", "hemoglobin_min",
    "creatinine_max", "bun_max", "lactate_max", "glucose_max", "sodium_min",
    "temperature_max", "heart_rate_mean", "resp_rate_mean", "mbp_min",
    "propofol_24h", "dexmed_24h", "vaso_24h",
    "htn", "diabetes", "ckd", "esrd_dialysis", "epilepsy_hx",
    "alcohol_abuse", "chf", "malignancy", "lupus_nephritis_icd", "hcq",
]
FEATS = [f for f in FEATS if f in dev.columns and f in val.columns]
LABEL = {
    "age": "年龄", "female": "女性", "sofa_noncns": "非神经 SOFA", "gcs_min_24h": "最低 GCS",
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
print(f"验证队列 (eICU-CRD): N={len(val):4d}  事件={val.y.sum():3d} ({val.y.mean()*100:.1f}%)")
print(f"共有候选特征: {len(FEATS)} 个")
print("=" * 66)

Xd_raw, yd = dev[FEATS].astype(float), dev.y.values
Xv_raw, yv = val[FEATS].astype(float), val.y.values

# 缺失率对比
mc = pd.DataFrame({
    "变量": [LABEL.get(f, f) for f in FEATS],
    "MIMIC 缺失%": (Xd_raw.isna().mean() * 100).round(1).values,
    "eICU 缺失%": (Xv_raw.isna().mean() * 100).round(1).values,
})
mc.to_csv(os.path.join(OUT, "feature_missingness_compare.csv"), index=False, encoding="utf-8-sig")
print("\n[两库缺失率 >20% 的特征]")
print(mc[(mc["MIMIC 缺失%"] > 20) | (mc["eICU 缺失%"] > 20)].to_string(index=False))

# 中位数填补（以开发集中位数为准，避免验证集信息泄漏）
med = Xd_raw.median()
Xd = Xd_raw.fillna(med)
Xv = Xv_raw.fillna(med)

# ============================================================
# 2. LASSO 特征选择（10 折 CV）
# ============================================================
sc0 = StandardScaler().fit(Xd)
las = LogisticRegressionCV(Cs=30, cv=StratifiedKFold(10, shuffle=True, random_state=SEED),
                           penalty="l1", solver="saga", scoring="roc_auc",
                           max_iter=10000, random_state=SEED, n_jobs=-1).fit(sc0.transform(Xd), yd)
coef = pd.DataFrame({"feature": FEATS, "label": [LABEL.get(f, f) for f in FEATS],
                     "coef": las.coef_[0]})
coef["abs"] = coef.coef.abs()
SEL = coef.loc[coef["abs"] > 1e-6, "feature"].tolist()
coef.sort_values("abs", ascending=False).to_csv(
    os.path.join(OUT, "ml_lasso_coefficients.csv"), index=False, encoding="utf-8-sig")
print(f"\n[LASSO] 最优 C={las.C_[0]:.4f}，选中 {len(SEL)}/{len(FEATS)} 个特征")
print(coef[coef["abs"] > 1e-6].sort_values("abs", ascending=False)
      [["label", "coef"]].round(3).to_string(index=False))

if len(SEL) < 5:
    SEL = coef.sort_values("abs", ascending=False).head(10).feature.tolist()
    print(f"[提示] LASSO 过于稀疏，回退为绝对系数前 10 名")

Xd_s, Xv_s = Xd[SEL], Xv[SEL]

# ============================================================
# 3. 模型定义
# ============================================================
cv10 = StratifiedKFold(10, shuffle=True, random_state=SEED)

models = {
    "Logistic 回归": Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
    "LASSO-LR": Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="saga", C=las.C_[0],
                                   max_iter=10000, random_state=SEED))]),
    "随机森林": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=600, max_depth=5,
                                       min_samples_leaf=8, max_features="sqrt",
                                       class_weight="balanced_subsample",
                                       random_state=SEED, n_jobs=-1))]),
    "XGBoost": Pipeline([
        ("clf", xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  reg_lambda=2.0, min_child_weight=5,
                                  eval_metric="logloss", random_state=SEED,
                                  n_jobs=-1))]),
}


def auc_ci(y, p, n_boot=2000, seed=SEED):
    """Bootstrap 计算 AUC 的 95% CI"""
    rng = np.random.default_rng(seed)
    a = roc_auc_score(y, p)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        bs.append(roc_auc_score(y[idx], p[idx]))
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def perf(y, p, thr=None):
    """综合性能指标"""
    a, lo, hi = auc_ci(y, p)
    if thr is None:   # Youden 最优阈值
        fpr, tpr, t = roc_curve(y, p)
        thr = t[np.argmax(tpr - fpr)]
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(AUC=a, AUC_lo=lo, AUC_hi=hi,
                Brier=brier_score_loss(y, p),
                AUPRC=average_precision_score(y, p),
                Sens=tp / (tp + fn) if tp + fn else np.nan,
                Spec=tn / (tn + fp) if tn + fp else np.nan,
                PPV=tp / (tp + fp) if tp + fp else np.nan,
                NPV=tn / (tn + fn) if tn + fn else np.nan,
                Acc=(tp + tn) / len(y), threshold=thr)


# ============================================================
# 4. 内部 10 折 CV + 外部验证
# ============================================================
res_rows, cv_preds, ext_preds, fitted = [], {}, {}, {}
for nm, mdl in models.items():
    pcv = cross_val_predict(mdl, Xd_s, yd, cv=cv10, method="predict_proba", n_jobs=-1)[:, 1]
    cv_preds[nm] = pcv
    m = mdl.fit(Xd_s, yd)
    fitted[nm] = m
    pex = m.predict_proba(Xv_s)[:, 1]
    ext_preds[nm] = pex

    r_cv = perf(yd, pcv)
    r_ex = perf(yv, pex)
    res_rows.append({
        "模型": nm,
        "内部 CV AUC (95% CI)": f"{r_cv['AUC']:.3f} ({r_cv['AUC_lo']:.3f}-{r_cv['AUC_hi']:.3f})",
        "内部 Brier": f"{r_cv['Brier']:.3f}",
        "内部 敏感度": f"{r_cv['Sens']:.3f}", "内部 特异度": f"{r_cv['Spec']:.3f}",
        "外部 AUC (95% CI)": f"{r_ex['AUC']:.3f} ({r_ex['AUC_lo']:.3f}-{r_ex['AUC_hi']:.3f})",
        "外部 Brier": f"{r_ex['Brier']:.3f}",
        "外部 敏感度": f"{r_ex['Sens']:.3f}", "外部 特异度": f"{r_ex['Spec']:.3f}",
        "_auc_cv": r_cv["AUC"], "_auc_ex": r_ex["AUC"],
    })
    print(f"  {nm:<14s} 内部 CV AUC={r_cv['AUC']:.3f}  外部 AUC={r_ex['AUC']:.3f}")

tab7 = pd.DataFrame(res_rows)
best = tab7.loc[tab7._auc_ex.idxmax(), "模型"]
tab7_out = tab7.drop(columns=["_auc_cv", "_auc_ex"])
tab7_out.to_csv(os.path.join(OUT, "table7_model_performance.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "table7_model_performance.md"), "w", encoding="utf-8") as f:
    f.write("# Table 7. 预测模型性能：内部交叉验证与外部验证\n\n")
    f.write(f"- 开发队列：MIMIC-IV，N={len(dev)}，CAM-ICU 谵妄 {yd.sum()} 例（{yd.mean()*100:.1f}%）\n")
    f.write(f"- 验证队列：eICU-CRD，N={len(val)}，急性脑功能障碍（诊断编码或 GCS≤12）"
            f"{yv.sum()} 例（{yv.mean()*100:.1f}%）\n")
    f.write(f"- 特征：LASSO 从 {len(FEATS)} 个候选中筛选出 {len(SEL)} 个\n")
    f.write(f"- 内部验证：10 折分层交叉验证；AUC 95% CI 由 2000 次 bootstrap 得到\n")
    f.write(f"- 敏感度/特异度基于各自 Youden 最优阈值\n\n")
    f.write(tab7_out.to_markdown(index=False))
    f.write(f"\n\n最佳外部验证性能模型：**{best}**\n")
    f.write("\n注：两队列结局定义不一致（MIMIC 为 CAM-ICU 量表，eICU 为诊断编码 + GCS），"
            "外部验证结果应解读为跨口径的可迁移性检验，而非严格意义的同质外推。\n")
print(f"\n[Table 7] 已保存。外部验证最佳模型：{best}")
print(tab7_out.to_string(index=False))

# ============================================================
# 5. ROC 曲线
# ============================================================
COLORS = {"Logistic 回归": "#185FA5", "LASSO-LR": "#0F6E56",
          "随机森林": "#993C1D", "XGBoost": "#534AB7"}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=160)
for ax, (preds, yy, ttl) in zip(axes, [(cv_preds, yd, f"内部 10 折交叉验证（MIMIC-IV, N={len(dev)}）"),
                                       (ext_preds, yv, f"外部验证（eICU-CRD, N={len(val)}）")]):
    for nm, p in preds.items():
        fpr, tpr, _ = roc_curve(yy, p)
        ax.plot(fpr, tpr, color=COLORS[nm], lw=1.9,
                label=f"{nm}  AUC={roc_auc_score(yy, p):.3f}")
    ax.plot([0, 1], [0, 1], color="#B4B2A9", lw=0.8, ls="--")
    ax.set_xlabel("1 − 特异度", fontsize=10.5)
    ax.set_ylabel("敏感度", fontsize=10.5)
    ax.set_title(ttl, fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
fig.suptitle("图 5. ROC 曲线：SLE 重症患者急性脑功能障碍预测模型", fontsize=12.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT, "fig_roc.png"), bbox_inches="tight")
print("[图] fig_roc.png")

# ============================================================
# 6. 校准曲线 + Hosmer-Lemeshow
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
    dof = max(1, len(gg) - 2)
    return float(chi2), float(stats.chi2.sf(chi2, dof))


fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=160)
cal_rows = []
for ax, (preds, yy, ttl) in zip(axes, [(cv_preds, yd, f"内部 10 折 CV（MIMIC-IV）"),
                                       (ext_preds, yv, f"外部验证（eICU-CRD）")]):
    for nm, p in preds.items():
        try:
            po, pp = calibration_curve(yy, p, n_bins=8, strategy="quantile")
            ax.plot(pp, po, "o-", color=COLORS[nm], lw=1.7, ms=4.5, label=nm)
        except Exception:
            pass
        chi2, ph = hl_test(yy, p)
        # 校准斜率与截距
        lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        import statsmodels.api as sm
        try:
            rr = sm.Logit(yy, sm.add_constant(lp)).fit(disp=0)
            slope, inter = rr.params[1], rr.params[0]
        except Exception:
            slope = inter = np.nan
        cal_rows.append(dict(队列="内部" if ttl.startswith("内部") else "外部", 模型=nm,
                             HL_chi2=chi2, HL_P=ph, 校准斜率=slope, 校准截距=inter,
                             Brier=brier_score_loss(yy, p)))
    ax.plot([0, 1], [0, 1], color="#B4B2A9", lw=0.9, ls="--", label="理想校准")
    ax.set_xlabel("预测概率", fontsize=10.5)
    ax.set_ylabel("实际发生率", fontsize=10.5)
    ax.set_title(ttl, fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.suptitle("图 6. 校准曲线", fontsize=12.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT, "fig_calibration.png"), bbox_inches="tight")

cal = pd.DataFrame(cal_rows).round(3)
cal.to_csv(os.path.join(OUT, "table8_calibration.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "table8_calibration.md"), "w", encoding="utf-8") as f:
    f.write("# Table 8. 模型校准度\n\n理想校准：斜率 = 1，截距 = 0，Hosmer-Lemeshow P > 0.05\n\n")
    f.write(cal.to_markdown(index=False))
print("[图] fig_calibration.png  [表] table8_calibration.md")
print(cal.to_string(index=False))

# ============================================================
# 7. 决策曲线分析 DCA
# ============================================================
def dca(y, p, thr):
    n = len(y)
    out = []
    prev = y.mean()
    for t in thr:
        yh = (p >= t).astype(int)
        tp = ((yh == 1) & (y == 1)).sum()
        fp = ((yh == 1) & (y == 0)).sum()
        nb = tp / n - (fp / n) * (t / (1 - t))
        nb_all = prev - (1 - prev) * (t / (1 - t))
        out.append((t, nb, nb_all))
    return np.array(out)


thr = np.linspace(0.01, 0.80, 160)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=160)
for ax, (preds, yy, ttl) in zip(axes, [(cv_preds, yd, "内部 10 折 CV（MIMIC-IV）"),
                                       (ext_preds, yv, "外部验证（eICU-CRD）")]):
    ref = None
    for nm, p in preds.items():
        a = dca(yy, p, thr)
        ax.plot(a[:, 0], a[:, 1], color=COLORS[nm], lw=1.8, label=nm)
        ref = a
    ax.plot(ref[:, 0], ref[:, 2], color="#5F5E5A", lw=1.2, ls="-.", label="全部干预")
    ax.axhline(0, color="#B4B2A9", lw=1.0, ls="--", label="全部不干预")
    ax.set_xlabel("阈概率", fontsize=10.5)
    ax.set_ylabel("净获益", fontsize=10.5)
    ax.set_title(ttl, fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.10, max(0.05, yy.mean() * 1.15))
    ax.set_xlim(0, 0.8)
fig.suptitle("图 7. 决策曲线分析（DCA）", fontsize=12.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT, "fig_dca.png"), bbox_inches="tight")
print("[图] fig_dca.png")

# ============================================================
# 8. SHAP 可解释性（XGBoost）
# ============================================================
try:
    import shap
    xgbm = fitted["XGBoost"].named_steps["clf"]
    Xs = Xd_s.copy()
    Xs.columns = [LABEL.get(c, c) for c in Xs.columns]
    expl = shap.TreeExplainer(xgbm)
    sv = expl.shap_values(Xs)
    # 二分类时 shap_values 可能返回 [class0, class1] 列表，统一取正类（index=1）
    if isinstance(sv, (list, tuple)):
        sv = sv[1] if len(sv) > 1 else sv[0]
    assert sv.ndim == 2, f"SHAP 值形状异常: {np.asarray(sv).shape}"

    plt.figure(figsize=(8.2, 6.2), dpi=160)
    shap.summary_plot(sv, Xs, show=False, max_display=15, plot_size=None)
    plt.title("图 8A. SHAP 特征重要性（蜂群图，XGBoost）", fontsize=11.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_shap_beeswarm.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.6, 5.6), dpi=160)
    shap.summary_plot(sv, Xs, plot_type="bar", show=False, max_display=15, plot_size=None)
    plt.title("图 8B. SHAP 平均绝对贡献度", fontsize=11.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_shap_bar.png"), bbox_inches="tight")
    plt.close()

    imp = pd.DataFrame({"特征": Xs.columns, "平均|SHAP|": np.abs(sv).mean(0)}) \
        .sort_values("平均|SHAP|", ascending=False)
    imp.round(4).to_csv(os.path.join(OUT, "shap_importance.csv"), index=False, encoding="utf-8-sig")
    print("[图] fig_shap_beeswarm.png / fig_shap_bar.png")
    print("\n[SHAP 特征重要性 Top 10]")
    print(imp.head(10).round(4).to_string(index=False))

    # 依赖图：SHAP 最重要的 4 个特征
    top4 = imp.head(4)["特征"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), dpi=160)
    for ax, ft in zip(axes.ravel(), top4):
        shap.dependence_plot(ft, sv, Xs, ax=ax, show=False, interaction_index=None,
                             color="#185FA5", alpha=0.6, dot_size=14)
        ax.set_title(ft, fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("图 9. SHAP 依赖图（Top 4 特征）", fontsize=12.5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUT, "fig_shap_dependence.png"), bbox_inches="tight")
    plt.close()
    print("[图] fig_shap_dependence.png")
except ImportError:
    print("[跳过 SHAP] shap 包未安装")

# ============================================================
# 9. 保存模型元信息
# ============================================================
meta = {
    "seed": SEED,
    "dev_cohort": {"source": "MIMIC-IV", "n": int(len(dev)), "events": int(yd.sum()),
                   "outcome": "CAM-ICU positive delirium"},
    "val_cohort": {"source": "eICU-CRD", "n": int(len(val)), "events": int(yv.sum()),
                   "outcome": "diagnosis code OR GCS<=12"},
    "candidate_features": FEATS,
    "selected_features": SEL,
    "lasso_C": float(las.C_[0]),
    "best_external_model": best,
    "performance": tab7_out.to_dict(orient="records"),
}
with open(os.path.join(OUT, "model_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# 保存预测概率供后续分析
pd.DataFrame({**{f"dev_{k}": v for k, v in cv_preds.items()}, "dev_y": yd}) \
    .to_csv(os.path.join(OUT, "pred_internal_cv.csv"), index=False)
pd.DataFrame({**{f"val_{k}": v for k, v in ext_preds.items()}, "val_y": yv}) \
    .to_csv(os.path.join(OUT, "pred_external.csv"), index=False)
print("\n[完成] 全部结果已保存至 output/")
