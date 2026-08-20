# -*- coding: utf-8 -*-
"""
09_core_firth.py  (v2 - 修正 Firth 实现)
针对"EPV 不足"局限的补强分析（对应 v14 稿件）：
  1) 构建预先设定的"简约核心模型"——取模型自身 LASSO 选出的前 9 个最强预测因子
     （剔除 GCS 以保持非循环，与主要外部验证口径一致），开发队列 EPV = 105/9 ≈ 11.7 ≥ 10。
  2) 自实现 Firth 惩罚逻辑回归（修正得分 U*_j = Σ_i x_ij [y_i - μ_i + h_ii(½-μ_i)]，
     h_ii 为 hat 矩阵对角元；IRLS 求解），与标准 Logistic（statsmodels）对照。
复现 04_ml_model.py 的开发队列定义：MIMIC-IV，cam_assessed==1 & primary_brain_injury==0。
"""
import os, json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from scipy import stats as st
warnings.filterwarnings("ignore")
SEED = 20260731
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

LABEL = {
    "age": "年龄", "female": "女性", "sofa_noncns": "非神经 SOFA", "gcs_min_24h": "最低 GCS",
    "sepsis3_24h": "脓毒症", "mech_vent_24h": "有创通气", "benzo_24h": "苯二氮䓬类",
    "opioid_24h": "阿片类", "steroid_24h": "糖皮质激素", "bicarbonate_min": "碳酸氢根最低",
    "lactate_max": "乳酸最高",
}

# ---------- 1. 复现开发队列 ----------
mim = pd.read_csv(os.path.join(ROOT, "data", "mimic_sle_analytic.csv"))
eic = pd.read_csv(os.path.join(ROOT, "data", "eicu_sle_analytic.csv"))
dev = mim[(mim.cam_assessed == 1) & (mim.primary_brain_injury == 0)].copy().reset_index(drop=True)
dev["sofa_noncns"] = dev["sofa_24h"] - dev["sofa_cns"]
dev["y"] = dev["cam_pos"].astype(int)
val = eic[(eic.primary_brain_injury == 0) & (eic.gcs_min_24h.notna())].copy().reset_index(drop=True)
val["y"] = val["abd_dx_gcs"].astype(int)
N, E = len(dev), int(dev.y.sum())
print(f"开发队列 N={N}  事件={E} ({dev.y.mean()*100:.1f}%)  外部(eICU, abd_dx_gcs) N={len(val)} 事件={int(val.y.sum())}")

FEATS = ["age","female","sofa_noncns","gcs_min_24h","sepsis3_24h","mech_vent_24h",
         "benzo_24h","opioid_24h","steroid_24h","bicarbonate_min","wbc_max","platelets_min",
         "hemoglobin_min","creatinine_max","bun_max","lactate_max","glucose_max","sodium_min",
         "temperature_max","heart_rate_mean","resp_rate_mean","mbp_min","propofol_24h","dexmed_24h",
         "vaso_24h","htn","diabetes","ckd","esrd_dialysis","epilepsy_hx","alcohol_abuse","chf",
         "malignancy","lupus_nephritis_icd","hcq"]
med = dev[FEATS].median()
Xd = dev[FEATS].fillna(med)
Xv = val[FEATS].fillna(med)

# ---------- 2. 简约核心模型（非循环，剔除 GCS）----------
CORE = ["sofa_noncns","mech_vent_24h","benzo_24h","opioid_24h","sepsis3_24h",
        "bicarbonate_min","age","female","steroid_24h"]
EPV = E / len(CORE)
print(f"核心模型预测因子数={len(CORE)}  EPV={EPV:.2f}  (>=10? {EPV>=10})")

Xc = Xd[CORE].copy()
Xc_val = Xv[CORE].copy()
y = dev.y.values

# ---------- 3. 标准 Logistic（statsmodels，逐单位 aOR）----------
smr = sm.Logit(y, sm.add_constant(Xc)).fit(disp=0)
std_beta = smr.params[1:]; std_se = smr.bse[1:]; std_p = smr.pvalues[1:]
std_aor = np.exp(std_beta); std_lo = np.exp(std_beta-1.96*std_se); std_hi = np.exp(std_beta+1.96*std_se)

# ---------- 4. Firth 惩罚逻辑回归（修正得分 IRLS，原始尺度）----------
def firth_fit(Xa, yy, max_iter=300, tol=1e-10):
    """Xa: design matrix WITH intercept column; yy: 0/1.
    Modified score U*_j = sum_i x_ij [y_i - mu_i + h_ii(1/2 - mu_i)] (Firth 1993; Heinze & Schemper)."""
    n, p = Xa.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = Xa @ beta
        mu = np.clip(1/(1+np.exp(-eta)), 1e-6, 1-1e-6)
        W = mu*(1-mu)
        I = Xa.T @ (W[:, None]*Xa)
        Iinv = np.linalg.inv(I + 1e-10*np.eye(p))
        h = W * np.einsum("ij,jk,ik->i", Xa, Iinv, Xa)     # hat 对角元 h_ii
        score = Xa.T @ (yy - mu) + Xa.T @ (h*(0.5 - mu))   # Firth 修正得分
        delta = np.linalg.solve(I + 1e-10*np.eye(p), score)
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            break
    mu = np.clip(1/(1+np.exp(-(Xa @ beta))), 1e-6, 1-1e-6)
    W = mu*(1-mu)
    I = Xa.T @ (W[:, None]*Xa)
    cov = np.linalg.inv(I + 1e-10*np.eye(p))
    return beta, np.sqrt(np.diag(cov))

Xa = sm.add_constant(Xc).values
fir_beta, fir_se = firth_fit(Xa, y)
fir_beta = fir_beta[1:]; fir_se = fir_se[1:]
fir_aor = np.exp(fir_beta); fir_lo = np.exp(fir_beta-1.96*fir_se); fir_hi = np.exp(fir_beta+1.96*fir_se)

# ---------- 5. 性能：10 折 CV AUC（标准 & Firth）----------
cv10 = StratifiedKFold(10, shuffle=True, random_state=SEED)
def auc_boot(yy, p, n_boot=2000):
    rng = np.random.default_rng(SEED); a = roc_auc_score(yy, p); bs=[]
    for _ in range(n_boot):
        idx = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[idx])) < 2: continue
        bs.append(roc_auc_score(yy[idx], p[idx]))
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

p_std_cv = cross_val_predict(Pipeline([("sc",StandardScaler()),
                  ("clf",LogisticRegression(max_iter=5000,random_state=SEED))]),
                 Xc, y, cv=cv10, method="predict_proba")[:,1]

p_fir_cv = np.zeros(len(y))
for tr, te in cv10.split(Xc, y):
    Xtr = sm.add_constant(Xc.iloc[tr]).values
    Xte = sm.add_constant(Xc.iloc[te]).values
    b, _ = firth_fit(Xtr, y[tr])
    p_fir_cv[te] = 1/(1+np.exp(-(Xte @ b)))

std_auc = auc_boot(y, p_std_cv); fir_auc = auc_boot(y, p_fir_cv)

def calib(yy, p, g=10):
    d = pd.DataFrame({"y":yy,"p":p})
    try: d["g"]=pd.qcut(d.p,g,labels=False,duplicates="drop")
    except ValueError: return (np.nan,)*4
    gg=d.groupby("g").agg(obs=("y","sum"),n=("y","size"),exp=("p","sum"))
    gg=gg[(gg.exp>0)&(gg.n-gg.exp>0)]
    chi2=(((gg.obs-gg.exp)**2)/(gg.exp*(1-gg.exp/gg.n))).sum(); dof=max(1,len(gg)-2)
    lp=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    rr=sm.Logit(yy,sm.add_constant(lp)).fit(disp=0)
    return float(chi2), float(st.chi2.sf(chi2,dof)), float(rr.params[1]), float(rr.params[0])
hl_std = calib(y, p_std_cv); hl_fir = calib(y, p_fir_cv)

# 外部（eICU, 非循环）AUC
Xa_full = sm.add_constant(Xc).values
b, _ = firth_fit(Xa_full, y)
Xev = sm.add_constant(Xc_val).values
p_ext = 1/(1+np.exp(-(Xev @ b)))
ext_auc = roc_auc_score(val.y.values, p_ext)
print(f"\n核心模型 eICU 外部（abd_dx_gcs, N={len(val)}, 事件={int(val.y.sum())}）AUC = {ext_auc:.3f}")

# 外部主要终点（abd_dx_only，扩大至不要求 GCS 的 256 例）
val2 = eic[(eic.primary_brain_injury == 0)].copy().reset_index(drop=True)
val2["y"] = val2["abd_dx_only"].astype(int)
Xc_val2 = val2[FEATS].fillna(med)[CORE].copy()
Xev2 = sm.add_constant(Xc_val2).values
p_ext2 = 1/(1+np.exp(-(Xev2 @ b)))
ext2_auc = roc_auc_score(val2.y.values, p_ext2)
print(f"核心模型 eICU 主要外部终点（abd_dx_only, N={len(val2)}, 事件={int(val2.y.sum())}）AUC = {ext2_auc:.3f}")

# ---------- 6. 汇总输出 ----------
rows = []
for j, f in enumerate(CORE):
    rows.append({
        "预测因子": LABEL.get(f, f), "变量": f,
        "标准 aOR": round(std_aor[j],3), "标准 95%CI": f"{std_lo[j]:.3f}–{std_hi[j]:.3f}", "标准 P": f"{std_p[j]:.3f}",
        "Firth aOR": round(fir_aor[j],3), "Firth 95%CI": f"{fir_lo[j]:.3f}–{fir_hi[j]:.3f}",
        "方向一致": "是" if (std_beta[j]>0)==(fir_beta[j]>0) else "否",
    })
coef_df = pd.DataFrame(rows)
coef_df.to_csv(os.path.join(OUT,"table_core_firth.csv"), index=False, encoding="utf-8-sig")
print("\n==== 核心模型系数（标准 vs Firth，逐单位 aOR）====")
print(coef_df.to_string(index=False))

perf = pd.DataFrame({
    "模型": ["标准 Logistic","Firth 惩罚"],
    "内部 CV AUC": [f"{std_auc[0]:.3f} ({std_auc[1]:.3f}–{std_auc[2]:.3f})",
                   f"{fir_auc[0]:.3f} ({fir_auc[1]:.3f}–{fir_auc[2]:.3f})"],
    "HL χ²": [f"{hl_std[0]:.2f}", f"{hl_fir[0]:.2f}"],
    "HL P": [f"{hl_std[1]:.3f}", f"{hl_fir[1]:.3f}"],
    "校准斜率": [f"{hl_std[2]:.3f}", f"{hl_fir[2]:.3f}"],
    "校准截距": [f"{hl_std[3]:.3f}", f"{hl_fir[3]:.3f}"],
})
perf.to_csv(os.path.join(OUT,"table_core_performance.csv"), index=False, encoding="utf-8-sig")
print("\n==== 性能 ====")
print(perf.to_string(index=False))
print(f"\n核心模型 eICU 非循环外部 AUC = {ext_auc:.3f}")
print(f"EPV = {EPV:.2f}  (开发队列 {N} 例 / {E} 事件 / {len(CORE)} 预测因子)")

meta = {"core_predictors":CORE,"EPV":round(EPV,2),"dev_N":N,"dev_events":E,
        "internal_cv_auc_std":[round(x,3) for x in std_auc],
        "internal_cv_auc_firth":[round(x,3) for x in fir_auc],
        "external_eicu_auc_firth":round(ext_auc,3),
        "external_dxonly_auc_firth":round(ext2_auc,3),
        "hl_std":[round(x,3) for x in hl_std],"hl_firth":[round(x,3) for x in hl_fir]}
json.dump(meta, open(os.path.join(OUT,"core_firth_meta.json"),"w"), ensure_ascii=False, indent=2)
print("\n[完成] 结果已保存 -> table_core_firth.csv / table_core_performance.csv / core_firth_meta.json")
