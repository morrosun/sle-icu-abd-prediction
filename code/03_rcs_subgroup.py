# -*- coding: utf-8 -*-
"""
03_rcs_subgroup.py
限制性立方样条(RCS)剂量-反应分析 + 亚组分析 + 敏感性分析

输入: data/main_analysis_set.csv (由 02_regression.py 生成)
输出:
  output/fig_rcs_continuous.png   连续变量 RCS 曲线(SOFA/GCS/年龄/体温)
  output/fig_rcs_steroid.png      激素剂量-谵妄 RCS
  output/table5_steroid_dose.csv/.md  激素分级 aOR
  output/table6_subgroup.csv/.md      亚组分析 + 交互检验
  output/fig_subgroup.png             亚组森林图
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from patsy import dmatrix, build_design_matrices

warnings.filterwarnings("ignore")
np.random.seed(20260730)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
d = pd.read_csv(os.path.join(ROOT, "data", "main_analysis_set.csv"))
Y = "cam_pos"
print(f"[主分析集] N={len(d)}  谵妄={int(d[Y].sum())} ({d[Y].mean()*100:.1f}%)")

LABEL = {
    "sofa_noncns": "非神经 SOFA 评分 (24h)",
    "gcs_min_24h": "24h 最低 GCS",
    "age": "年龄 (岁)",
    "temperature_max": "24h 最高体温 (℃)",
    "pred_eq_24h": "泼尼松当量日剂量 (mg)",
}
# Model 3 协变量（RCS 时用于校正，目标变量自身除外）
COV = ["age", "female", "sofa_noncns", "gcs_min_24h", "alcohol_abuse", "epilepsy_hx",
       "bicarbonate_min", "resp_rate_mean", "temperature_max",
       "sepsis3_24h", "mech_vent_24h", "benzo_24h", "opioid_24h", "steroid_24h"]

# 简单中位数填补（RCS 为探索性分析，与主分析 MICE 结果互为印证）
work = d.copy()
for c in set(COV + ["pred_eq_24h"]):
    if work[c].isna().any():
        work[c] = work[c].fillna(work[c].median())


def rcs_curve(df, xvar, covs, knots=4, xref=None, xgrid=None):
    """校正协变量后的 RCS 剂量-反应曲线（相对 xref 的 OR 及 95% CI，delta 法）"""
    covs = [c for c in covs if c != xvar]
    x = df[xvar].astype(float)
    # constraints='center' → 生成 df-1 个基函数，与截距不共线（等价于 R rms::rcs）
    FORM = f"cr(x, df={knots}, constraints='center')"
    di = dmatrix(FORM, {"x": x}, return_type="dataframe")
    di.columns = [f"s{i}" for i in range(di.shape[1])]
    di = di.drop(columns=["s0"])                      # 去掉 patsy 自带的 Intercept 列
    X = pd.concat([di.reset_index(drop=True),
                   df[covs].astype(float).reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    res = sm.Logit(df[Y].values, X).fit(disp=0, maxiter=300)

    if xgrid is None:
        xgrid = np.linspace(x.quantile(0.01), x.quantile(0.99), 120)
    if xref is None:
        xref = float(x.median())
    dinfo = dmatrix(FORM, {"x": x}, return_type="dataframe").design_info
    bg = np.asarray(build_design_matrices([dinfo], {"x": xgrid})[0])[:, 1:]
    br = np.asarray(build_design_matrices([dinfo], {"x": np.array([xref])})[0])[:, 1:]
    idx = [X.columns.get_loc(c) for c in di.columns]
    C = np.zeros((len(xgrid), X.shape[1]))
    C[:, idx] = bg - br
    beta = res.params.values
    V = res.cov_params().values
    logor = C @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", C, V, C))

    # 非线性检验：样条高阶项联合为 0
    if len(idx) > 1:
        R = np.zeros((len(idx) - 1, X.shape[1]))
        for i, j in enumerate(idx[1:]):
            R[i, j] = 1
        wald = res.wald_test(R, scalar=True)
        p_nl = float(wald.pvalue)
    else:
        p_nl = np.nan
    # 总体关联检验
    R0 = np.zeros((len(idx), X.shape[1]))
    for i, j in enumerate(idx):
        R0[i, j] = 1
    p_all = float(res.wald_test(R0, scalar=True).pvalue)
    return xgrid, np.exp(logor), np.exp(logor - 1.96 * se), np.exp(logor + 1.96 * se), \
        p_all, p_nl, xref


# ---------------- 图1：连续变量 RCS ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

targets = ["sofa_noncns", "gcs_min_24h", "age", "temperature_max"]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), dpi=160)
rcs_summary = []
for ax, v in zip(axes.ravel(), targets):
    refv = {"sofa_noncns": 2, "gcs_min_24h": 15, "age": 50, "temperature_max": 37.0}[v]
    xg, orr, lo, hi, p_all, p_nl, xref = rcs_curve(work, v, COV, knots=4, xref=refv)
    ax.fill_between(xg, lo, hi, color="#B5D4F4", alpha=0.45, lw=0)
    ax.plot(xg, orr, color="#185FA5", lw=2)
    ax.axhline(1, color="#888780", lw=0.8, ls="--")
    ax.axvline(xref, color="#B4B2A9", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(LABEL[v], fontsize=10)
    ax.set_ylabel("调整后 OR", fontsize=10)
    ax.set_title(f"P(总体)={'<0.001' if p_all<0.001 else f'{p_all:.3f}'}  |  "
                 f"P(非线性)={'<0.001' if p_nl<0.001 else f'{p_nl:.3f}'}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(max(0.05, lo.min() * 0.8), min(50, hi.max() * 1.2))
    rcs_summary.append(dict(variable=v, label=LABEL[v], ref=xref,
                            P_overall=p_all, P_nonlinear=p_nl))
fig.suptitle(f"图 2. 连续变量与 ICU 谵妄的剂量-反应关系（限制性立方样条，4 节点，N={len(work)}）",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "fig_rcs_continuous.png"), bbox_inches="tight")
print("\n[RCS 连续变量]")
print(pd.DataFrame(rcs_summary).round(4).to_string(index=False))

# ---------------- 图2 + Table5：激素剂量 ----------------
COV_ST = [c for c in COV if c != "steroid_24h"]

# (a) 分级 OR（none 为参照）
work["st_low"] = (work.steroid_cat == "low(<30)").astype(int)
work["st_mod"] = (work.steroid_cat == "moderate(30-100)").astype(int)
work["st_hi"] = (work.steroid_cat == "high(100-250)").astype(int)
work["st_pulse"] = (work.steroid_cat == "pulse(>=250)").astype(int)
stv = ["st_low", "st_mod", "st_hi", "st_pulse"]
Xst = sm.add_constant(work[stv + COV_ST].astype(float))
rst = sm.Logit(work[Y].values, Xst).fit(disp=0, maxiter=300)
rows = []
cnt = work.steroid_cat.value_counts()
pos = work.groupby("steroid_cat")[Y].sum()
order = [("none", None, "未使用（参照）"), ("low(<30)", "st_low", "<30 mg/d"),
         ("moderate(30-100)", "st_mod", "30–100 mg/d"),
         ("high(100-250)", "st_hi", "100–250 mg/d"),
         ("pulse(>=250)", "st_pulse", "≥250 mg/d（冲击）")]
for cat, vv, lab in order:
    n = int(cnt.get(cat, 0)); e = int(pos.get(cat, 0))
    if vv is None:
        rows.append(dict(分组=lab, 例数=n, 谵妄=f"{e} ({e/n*100:.1f}%)" if n else "-",
                         **{"aOR (95% CI)": "1.00 (参照)", "P": "—"}))
    else:
        b, s, p = rst.params[vv], rst.bse[vv], rst.pvalues[vv]
        rows.append(dict(分组=lab, 例数=n, 谵妄=f"{e} ({e/n*100:.1f}%)" if n else "-",
                         **{"aOR (95% CI)": f"{np.exp(b):.2f} ({np.exp(b-1.96*s):.2f}-{np.exp(b+1.96*s):.2f})",
                            "P": "<0.001" if p < 0.001 else f"{p:.3f}"}))
# 趋势检验（把分级编码为 0-4 的序数变量）
work["st_ord"] = work.steroid_cat.map({"none": 0, "low(<30)": 1, "moderate(30-100)": 2,
                                       "high(100-250)": 3, "pulse(>=250)": 4}).astype(float)
rtr = sm.Logit(work[Y].values, sm.add_constant(work[["st_ord"] + COV_ST].astype(float))).fit(disp=0)
p_trend = rtr.pvalues["st_ord"]
tab5 = pd.DataFrame(rows)
tab5.to_csv(os.path.join(OUT, "table5_steroid_dose.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "table5_steroid_dose.md"), "w", encoding="utf-8") as f:
    f.write("# Table 5. 糖皮质激素日剂量（泼尼松当量）与 ICU 谵妄\n\n")
    f.write(f"主分析集 N={len(work)}。校正 Model 3 协变量（年龄、性别、非神经 SOFA、最低 GCS、"
            "酒精滥用、癫痫史、碳酸氢根、呼吸频率、最高体温、脓毒症、机械通气、苯二氮卓、阿片类）。\n\n")
    f.write(tab5.to_markdown(index=False))
    f.write(f"\n\n趋势性检验 P = {p_trend:.3f}\n")
print("\n[Table 5 激素剂量分级]")
print(tab5.to_string(index=False))
print(f"趋势 P = {p_trend:.3f}")

# (b) 在使用激素者中做 RCS
users = work[work.pred_eq_24h > 0].copy().reset_index(drop=True)
print(f"\n[激素使用者亚组] n={len(users)}  谵妄={int(users[Y].sum())}")
fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
if len(users) >= 60 and users[Y].sum() >= 20:
    ref_dose = float(users.pred_eq_24h.median())
    xg, orr, lo, hi, p_all, p_nl, xref = rcs_curve(
        users, "pred_eq_24h", [c for c in COV_ST if users[c].nunique() > 1],
        knots=3, xref=ref_dose,
        xgrid=np.linspace(users.pred_eq_24h.quantile(0.02),
                          users.pred_eq_24h.quantile(0.95), 120))
    ax.fill_between(xg, lo, hi, color="#F5C4B3", alpha=0.5, lw=0)
    ax.plot(xg, orr, color="#993C1D", lw=2)
    ax.axhline(1, color="#888780", lw=0.8, ls="--")
    ax.axvline(xref, color="#B4B2A9", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_title(f"激素使用者中剂量-谵妄关系（n={len(users)}）\n"
                 f"P(总体)={p_all:.3f}  P(非线性)={p_nl:.3f}  参照={xref:.0f} mg", fontsize=10)
    ax.set_xlabel("泼尼松当量日剂量 (mg)", fontsize=10)
    ax.set_ylabel("调整后 OR", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    print(f"[RCS 激素] P(总体)={p_all:.3f}  P(非线性)={p_nl:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_rcs_steroid.png"), bbox_inches="tight")

# ---------------- Table6 + 图3：亚组分析 ----------------
EXPO = "benzo_24h"
EXPO_LAB = "苯二氮卓类 (24h)"
SUB = [
    ("age_grp", "年龄", lambda x: np.where(x.age >= 60, "≥60 岁", "<60 岁")),
    ("sex_grp", "性别", lambda x: np.where(x.female == 1, "女性", "男性")),
    ("sep_grp", "脓毒症", lambda x: np.where(x.sepsis3_24h == 1, "有", "无")),
    ("ln_grp", "狼疮肾炎", lambda x: np.where(x.lupus_nephritis_icd == 1, "有", "无")),
    ("vent_grp", "有创通气", lambda x: np.where(x.mech_vent_24h == 1, "有", "无")),
    ("st_grp", "ICU 激素", lambda x: np.where(x.steroid_24h == 1, "有", "无")),
    ("sofa_grp", "非神经 SOFA", lambda x: np.where(x.sofa_noncns >= 4, "≥4", "<4")),
]
ADJ = ["age", "female", "sofa_noncns", "gcs_min_24h", "sepsis3_24h", "mech_vent_24h"]
rows, plotd = [], []
for key, name, fn in SUB:
    work[key] = fn(work)
    lv = sorted(work[key].unique())
    rows.append({"亚组": f"**{name}**", "例数": "", "谵妄率": "",
                 "aOR (95% CI)": "", "P": "", "交互 P": ""})
    ors = []
    for l in lv:
        s = work[work[key] == l]
        adj = [c for c in ADJ if s[c].nunique() > 1 and c != EXPO]
        try:
            r = sm.Logit(s[Y].values,
                         sm.add_constant(s[[EXPO] + adj].astype(float))).fit(disp=0, maxiter=300)
            b, se_, p = r.params[EXPO], r.bse[EXPO], r.pvalues[EXPO]
            o, loo, hii = np.exp(b), np.exp(b - 1.96 * se_), np.exp(b + 1.96 * se_)
            txt = f"{o:.2f} ({loo:.2f}-{hii:.2f})"
            ptxt = "<0.001" if p < 0.001 else f"{p:.3f}"
        except Exception:
            o = loo = hii = np.nan; txt, ptxt = "不可估计", "—"
        rows.append({"亚组": f"　{l}", "例数": len(s),
                     "谵妄率": f"{s[Y].sum()}/{len(s)} ({s[Y].mean()*100:.1f}%)",
                     "aOR (95% CI)": txt, "P": ptxt, "交互 P": ""})
        ors.append((f"{name}: {l}", o, loo, hii, len(s)))
    # 交互检验
    try:
        w2 = work.copy()
        w2["_g"] = (w2[key] == lv[1]).astype(float)
        w2["_i"] = w2["_g"] * w2[EXPO]
        adj2 = [c for c in ADJ if c != EXPO]
        ri = sm.Logit(w2[Y].values,
                      sm.add_constant(w2[[EXPO, "_g", "_i"] + adj2].astype(float))).fit(disp=0, maxiter=300)
        pint = ri.pvalues["_i"]
        rows[-len(lv) - 1]["交互 P"] = "<0.001" if pint < 0.001 else f"{pint:.3f}"
    except Exception:
        rows[-len(lv) - 1]["交互 P"] = "—"
    plotd += ors

tab6 = pd.DataFrame(rows)
tab6.to_csv(os.path.join(OUT, "table6_subgroup.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "table6_subgroup.md"), "w", encoding="utf-8") as f:
    f.write(f"# Table 6. 亚组分析：{EXPO_LAB}与 ICU 谵妄的关联\n\n")
    f.write(f"主分析集 N={len(work)}。各亚组内校正年龄、性别、非神经 SOFA、最低 GCS、脓毒症、机械通气。\n\n")
    f.write(tab6.to_markdown(index=False))
print("\n[Table 6 亚组分析]")
print(tab6.to_string(index=False))

# 亚组森林图
pf = pd.DataFrame(plotd, columns=["label", "OR", "lo", "hi", "n"]).dropna()
fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(pf) + 1.8), dpi=160)
yp = np.arange(len(pf))[::-1]
ax.errorbar(pf.OR, yp, xerr=[pf.OR - pf.lo, pf.hi - pf.OR], fmt="s",
            color="#993C1D", ecolor="#F0997B", elinewidth=1.5, capsize=3, markersize=5)
ax.axvline(1, color="#888780", lw=0.8, ls="--")
ax.set_yticks(yp); ax.set_yticklabels(pf.label, fontsize=9.5)
ax.set_xscale("log"); ax.set_xlabel("调整后 OR (95% CI，对数刻度)", fontsize=10)
ax.set_title(f"图 4. 亚组分析：{EXPO_LAB}与 ICU 谵妄", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
for i, (_, r) in zip(yp, pf.iterrows()):
    ax.text(1.02, i, f"  {r.OR:.2f} ({r.lo:.2f}-{r.hi:.2f})  n={int(r.n)}",
            transform=ax.get_yaxis_transform(), va="center", fontsize=8.5)
ax.set_xlim(max(0.1, pf.lo.min() * 0.7), pf.hi.max() * 2.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_subgroup.png"), bbox_inches="tight")
print("\n[完成] 图表已保存至 output/")
