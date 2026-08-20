# -*- coding: utf-8 -*-
"""
02_regression.py
MIMIC-IV SLE ICU 队列 —— 急性脑功能障碍(CAM-ICU 谵妄)危险因素分析

主分析集: cam_assessed==1 且 primary_brain_injury==0  (N=264, 事件105)
建模口径: 仅使用入 ICU 24h 内可得变量; 剔除反向因果变量(氟哌啶醇/非典型抗精神病药)

流程:
  A. 缺失情况汇总
  B. 单变量 logistic (完整病例)
  C. 变量筛选 + 共线性(VIF)
  D. 多变量 logistic (MICE m=5, Rubin 合并) + 完整病例敏感性分析
输出:
  output/table2_univariate.csv/.md
  output/table3_multivariable.csv/.md
  output/missingness.csv
  output/fig_forest.png
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.imputation.mice import MICEData

warnings.filterwarnings("ignore")
np.random.seed(20260730)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "mimic_sle_analytic.csv")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

# ---------------- 主分析集 ----------------
main = df[(df.cam_assessed == 1) & (df.primary_brain_injury == 0)].copy().reset_index(drop=True)
Y = "cam_pos"
print(f"[主分析集] N={len(main)}  谵妄={int(main[Y].sum())} ({main[Y].mean()*100:.1f}%)")

# 非神经 SOFA（剔除 CNS 分量，避免与结局定义重叠）
main["sofa_noncns"] = main["sofa_24h"] - main["sofa_cns"]

# 种族哑变量（White 为参照）
main["race_black"] = (main.race_grp == "Black").astype(int)
main["race_other"] = (~main.race_grp.isin(["White", "Black"])).astype(int)

# 激素分级哑变量（none 为参照）
main["st_low"] = (main.steroid_cat == "low(<30)").astype(int)
main["st_mod"] = (main.steroid_cat == "moderate(30-100)").astype(int)
main["st_high"] = (main.steroid_cat.isin(["high(100-250)", "pulse(>=250)"])).astype(int)

# ---------------- 候选变量 ----------------
CONT = {
    "age": "年龄, 岁",
    "sofa_noncns": "非神经 SOFA (24h)",
    "charlson": "Charlson 合并症指数",
    "gcs_min_24h": "GCS 最低值 (24h)",
    "heart_rate_mean": "心率均值, 次/分",
    "mbp_min": "平均动脉压最低, mmHg",
    "resp_rate_mean": "呼吸频率均值, 次/分",
    "temperature_max": "体温最高, ℃",
    "spo2_min": "SpO2 最低, %",
    "wbc_max": "白细胞最高, K/uL",
    "hemoglobin_min": "血红蛋白最低, g/dL",
    "platelets_min": "血小板最低, K/uL",
    "creatinine_max": "肌酐最高, mg/dL",
    "bun_max": "尿素氮最高, mg/dL",
    "sodium_min": "血钠最低, mmol/L",
    "bicarbonate_min": "碳酸氢根最低, mmol/L",
    "aniongap_max": "阴离子间隙最高, mmol/L",
    "glucose_max": "血糖最高, mg/dL",
    "inr_max": "INR 最高",
    "abs_lymphocytes_min": "淋巴细胞绝对值最低, K/uL",
    "lactate_max": "乳酸最高, mmol/L",
    "uo_24h": "24h 尿量, mL",
    "pred_eq_24h": "泼尼松当量 (24h), mg",
}
BIN = {
    "female": "女性",
    "race_black": "黑人 (参照: 白人)",
    "race_other": "其他种族 (参照: 白人)",
    "lupus_nephritis_icd": "狼疮肾炎 (ICD)",
    "sle_organ_involv": "SLE 其他脏器受累 (ICD)",
    "htn": "高血压",
    "diabetes": "糖尿病",
    "ckd": "慢性肾脏病",
    "esrd_dialysis": "终末期肾病/透析",
    "epilepsy_hx": "癫痫病史",
    "dementia_hx": "痴呆",
    "depression_hx": "抑郁/情感障碍",
    "alcohol_abuse": "酒精滥用",
    "aps_antiphospholipid": "抗磷脂综合征",
    "chf": "心力衰竭",
    "copd": "慢性阻塞性肺病",
    "malignancy": "恶性肿瘤",
    "sepsis3_24h": "脓毒症 (Sepsis-3, 24h)",
    "mech_vent_24h": "有创机械通气 (24h)",
    "vaso_24h": "血管活性药 (24h)",
    "rrt_24h": "肾脏替代治疗 (24h)",
    "steroid_24h": "ICU 24h 内使用激素",
    "st_low": "激素 <30mg/d (参照: 未用)",
    "st_mod": "激素 30-100mg/d (参照: 未用)",
    "st_high": "激素 >=100mg/d (参照: 未用)",
    "hcq": "羟氯喹",
    "mmf": "霉酚酸酯",
    "aza": "硫唑嘌呤",
    "cni": "钙调磷酸酶抑制剂",
    "benzo_24h": "苯二氮卓类 (24h)",
    "propofol_24h": "丙泊酚 (24h)",
    "dexmed_24h": "右美托咪定 (24h)",
    "opioid_24h": "阿片类 (24h)",
}
LABEL = {**CONT, **BIN}
ALL_VARS = list(CONT) + list(BIN)

# ---------------- A. 缺失情况 ----------------
miss = pd.DataFrame({
    "variable": ALL_VARS,
    "label": [LABEL[v] for v in ALL_VARS],
    "missing_n": [int(main[v].isna().sum()) for v in ALL_VARS],
})
miss["missing_pct"] = (miss.missing_n / len(main) * 100).round(1)
miss = miss.sort_values("missing_pct", ascending=False)
miss.to_csv(os.path.join(OUT, "missingness.csv"), index=False, encoding="utf-8-sig")
print("\n[缺失率 >10% 的变量]")
print(miss[miss.missing_pct > 10].to_string(index=False))

HIGH_MISS = set(miss.loc[miss.missing_pct > 50, "variable"])   # 主模型排除
print(f"\n[缺失>50%, 主模型排除] {sorted(HIGH_MISS) if HIGH_MISS else '无'}")


# ---------------- 工具函数 ----------------
def fit_logit(y, X):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.Logit(y, Xc).fit(disp=0, maxiter=200)


def or_row(res, name):
    b = res.params[name]; se = res.bse[name]; p = res.pvalues[name]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p


# ---------------- B. 单变量 logistic ----------------
rows = []
for v in ALL_VARS:
    sub = main[[Y, v]].dropna()
    if sub[v].nunique() < 2 or len(sub) < 30:
        continue
    if v in BIN and (sub[v].sum() < 5 or (len(sub) - sub[v].sum()) < 5):
        continue
    scale, unit = 1.0, ""
    if v in CONT:
        # 对量纲大的连续变量按标准差或固定单位缩放，使 OR 可解释
        if v == "uo_24h":
            scale, unit = 500.0, " (每 500 mL)"
        elif v == "glucose_max":
            scale, unit = 10.0, " (每 10 mg/dL)"
        elif v == "pred_eq_24h":
            scale, unit = 10.0, " (每 10 mg)"
        elif v == "platelets_min":
            scale, unit = 50.0, " (每 50 K/uL)"
        elif v == "age":
            scale, unit = 10.0, " (每 10 岁)"
    x = sub[[v]].astype(float) / scale
    try:
        res = fit_logit(sub[Y].values, x)
        o, lo, hi, p = or_row(res, v)
        rows.append(dict(variable=v, label=LABEL[v] + unit, n=len(sub),
                         OR=o, lo=lo, hi=hi, P=p))
    except Exception as e:
        print(f"  ! 单变量失败 {v}: {e}")

uni = pd.DataFrame(rows).sort_values("P")
uni["OR (95% CI)"] = uni.apply(lambda r: f"{r.OR:.2f} ({r.lo:.2f}-{r.hi:.2f})", axis=1)
uni["P_fmt"] = uni.P.apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
uni_out = uni[["label", "n", "OR (95% CI)", "P_fmt"]].rename(
    columns={"label": "变量", "n": "可分析例数", "P_fmt": "P"})
uni_out.to_csv(os.path.join(OUT, "table2_univariate.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "table2_univariate.md"), "w", encoding="utf-8") as f:
    f.write("# Table 2. 单变量 logistic 回归：ICU 谵妄 (CAM-ICU) 的危险因素\n\n")
    f.write(f"主分析集 N={len(main)}，谵妄 {int(main[Y].sum())} 例 "
            f"({main[Y].mean()*100:.1f}%)。已排除原发急性脑损伤。完整病例分析。\n\n")
    f.write(uni_out.to_markdown(index=False))
print(f"\n[单变量] 完成 {len(uni)} 个变量，P<0.05 共 {int((uni.P<0.05).sum())} 个")
print(uni_out.head(20).to_string(index=False))

# ---------------- C. 变量筛选 + 共线性 ----------------
CAND = [v for v in uni.loc[uni.P < 0.10, "variable"] if v not in HIGH_MISS]
FORCE = ["age", "female"]
for v in FORCE:
    if v not in CAND and v not in HIGH_MISS:
        CAND.append(v)
# 强制保留临床关注的激素暴露
if "steroid_24h" not in CAND:
    CAND.append("steroid_24h")
print(f"\n[候选入模变量 n={len(CAND)}] {CAND}")

# VIF（完整病例）
cc = main[CAND].dropna()
vif = pd.DataFrame({
    "variable": CAND,
    "label": [LABEL[v] for v in CAND],
    "VIF": [variance_inflation_factor(sm.add_constant(cc.astype(float)).values, i + 1)
            for i in range(len(CAND))]
}).sort_values("VIF", ascending=False)
print("\n[VIF]")
print(vif.round(2).to_string(index=False))
DROP_VIF = list(vif.loc[vif.VIF > 5, "variable"])
if DROP_VIF:
    print(f"[VIF>5 剔除] {DROP_VIF}")
CAND = [v for v in CAND if v not in DROP_VIF]
n_ev = int(main[Y].sum())
max_p = max(6, n_ev // 10)   # EPV>=10 约束

# ---------------- D. MICE 插补 (m=5) ----------------
M = 5
IMP_VARS = sorted(set(CAND) | {Y})
imp_src = main[IMP_VARS].astype(float).copy()
imp_sets = []
if imp_src.drop(columns=[Y]).isna().any().any():
    mice = MICEData(imp_src)
    mice.set_imputer(Y, formula=" + ".join(CAND))
    for i in range(M + 5):                 # 前 5 轮 burn-in
        mice.update_all(1)
        if i >= 5:
            d = mice.data.copy()
            d[Y] = main[Y].values          # 结局不插补，还原真实值
            imp_sets.append(d)
        if len(imp_sets) >= M:
            break
    print(f"\n[MICE] 已生成 {len(imp_sets)} 套插补数据集 "
          f"(插补变量: {[v for v in CAND if main[v].isna().any()]})")
else:
    imp_sets = [main[IMP_VARS].astype(float).copy()] * M
    print("\n[MICE] 候选变量无缺失，跳过插补")

# ---------------- D1. LASSO 变量选择（每套插补集，多数投票） ----------------
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sel_count = pd.Series(0, index=CAND, dtype=int)
for d in imp_sets:
    Xs = StandardScaler().fit_transform(d[CAND].values)
    las = LogisticRegressionCV(Cs=25, cv=10, penalty="l1", solver="saga",
                               scoring="neg_log_loss", max_iter=8000,
                               random_state=42, n_jobs=-1).fit(Xs, d[Y].astype(int).values)
    sel_count[np.array(CAND)[np.abs(las.coef_[0]) > 1e-6]] += 1

sel_tab = pd.DataFrame({"variable": CAND, "label": [LABEL[v] for v in CAND],
                        "selected_in_m": sel_count.values,
                        "uni_P": [uni.set_index("variable").P.get(v, np.nan) for v in CAND]}) \
    .sort_values(["selected_in_m", "uni_P"], ascending=[False, True])
print("\n[LASSO 变量选择：在 m=5 套插补集中被选中的次数]")
print(sel_tab.to_string(index=False))
sel_tab.to_csv(os.path.join(OUT, "lasso_selection.csv"), index=False, encoding="utf-8-sig")

FORCE_KEEP = ["age", "female", "steroid_24h"]     # 临床强制保留
picked = [v for v in sel_tab.loc[sel_tab.selected_in_m >= 3, "variable"] if v not in FORCE_KEEP]
FINAL = FORCE_KEEP + picked[: max(0, max_p - len(FORCE_KEEP))]
FINAL = [v for v in CAND if v in FINAL]           # 保持原顺序
print(f"\n[最终多变量模型 n={len(FINAL)}, EPV={n_ev/len(FINAL):.1f}] {FINAL}")

# ---------------- D2. 多变量 logistic：MICE + Rubin 合并 ----------------
def pool_mice(varlist):
    """在 m 套插补数据集上拟合 logistic，按 Rubin 规则合并"""
    cf, vr = [], []
    for dd in imp_sets:
        rr = fit_logit(dd[Y].astype(int).values, dd[varlist].astype(float))
        cf.append(rr.params.values); vr.append(rr.bse.values ** 2)
    cf = np.array(cf); vr = np.array(vr)
    q, u = cf.mean(0), vr.mean(0)
    Bv = cf.var(0, ddof=1)
    if np.allclose(Bv, 0):
        s = np.sqrt(u); p = 2 * stats.norm.sf(np.abs(q / s))
    else:
        s = np.sqrt(u + (1 + 1 / M) * Bv)
        dfr = np.clip((M - 1) * (1 + u / ((1 + 1 / M) * Bv + 1e-12)) ** 2, 1, 1e6)
        p = 2 * stats.t.sf(np.abs(q / s), dfr)
    out = pd.DataFrame({"variable": ["const"] + varlist, "beta": q, "se": s, "P": p})
    out["OR"] = np.exp(out.beta)
    out["lo"] = np.exp(out.beta - 1.96 * out.se)
    out["hi"] = np.exp(out.beta + 1.96 * out.se)
    au = np.mean([roc_auc_score(dd[Y], fit_logit(dd[Y].astype(int).values,
                  dd[varlist].astype(float)).predict(
                  sm.add_constant(dd[varlist].astype(float)))) for dd in imp_sets])
    return out[out.variable != "const"].reset_index(drop=True), au


mi_res, _auc_final = pool_mice(FINAL)
qbar = np.r_[np.nan, mi_res.beta.values]  # 兼容后续引用

# 完整病例敏感性分析
cc2 = main[[Y] + FINAL].dropna()
res_cc = fit_logit(cc2[Y].values, cc2[FINAL].astype(float))
print(f"[完整病例多变量] n={len(cc2)}  Pseudo R2={res_cc.prsquared:.3f}")

mi_res["label"] = mi_res.variable.map(LABEL)

# 完整病例结果并列
cc_map = {v: or_row(res_cc, v) for v in FINAL}
mi_res["OR_cc"] = mi_res.variable.map(lambda v: cc_map[v][0])
mi_res["lo_cc"] = mi_res.variable.map(lambda v: cc_map[v][1])
mi_res["hi_cc"] = mi_res.variable.map(lambda v: cc_map[v][2])
mi_res["P_cc"] = mi_res.variable.map(lambda v: cc_map[v][3])

mi_res = mi_res.sort_values("P").reset_index(drop=True)
tab3 = pd.DataFrame({
    "变量": mi_res.label,
    "aOR (95% CI) [MICE]": mi_res.apply(lambda r: f"{r.OR:.2f} ({r.lo:.2f}-{r.hi:.2f})", axis=1),
    "P [MICE]": mi_res.P.apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"),
    "aOR (95% CI) [完整病例]": mi_res.apply(lambda r: f"{r.OR_cc:.2f} ({r.lo_cc:.2f}-{r.hi_cc:.2f})", axis=1),
    "P [完整病例]": mi_res.P_cc.apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"),
})

# 表观 AUC（MICE 各插补集平均 + 完整病例）
aucs = []
for d in imp_sets:
    r = fit_logit(d[Y].astype(int).values, d[FINAL].astype(float))
    aucs.append(roc_auc_score(d[Y], r.predict(sm.add_constant(d[FINAL].astype(float)))))
auc_mi = float(np.mean(aucs))
auc_cc = roc_auc_score(cc2[Y], res_cc.predict(sm.add_constant(cc2[FINAL].astype(float))))
print(f"\n[表观 AUC] MICE 平均 {auc_mi:.3f} | 完整病例 {auc_cc:.3f}")

tab3.to_csv(os.path.join(OUT, "table3_multivariable.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "table3_multivariable.md"), "w", encoding="utf-8") as f:
    f.write("# Table 3. 多变量 logistic 回归：ICU 谵妄 (CAM-ICU) 的独立危险因素\n\n")
    f.write(f"主分析集 N={len(main)}（已排除原发急性脑损伤），事件 {n_ev} 例（{n_ev/len(main)*100:.1f}%）。\n\n")
    f.write(f"- 变量选择：单变量 P<0.10 → VIF>5 剔除 → 在 m={M} 套多重插补数据集上分别做 "
            f"10 折交叉验证 LASSO，被选中 ≥3/5 次者入模（年龄、性别、ICU 24h 激素强制保留）\n")
    f.write(f"- 主结果：多重插补 m={M}，Rubin 规则合并；敏感性分析：完整病例 n={len(cc2)}\n")
    f.write(f"- 每变量事件数 EPV = {n_ev/len(FINAL):.1f}\n")
    f.write(f"- 模型表观 AUC = {auc_mi:.3f}（MICE 平均）/ {auc_cc:.3f}（完整病例）；"
            f"Pseudo R² = {res_cc.prsquared:.3f}\n\n")
    f.write(tab3.to_markdown(index=False))
    f.write("\n\n注：氟哌啶醇与非典型抗精神病药为谵妄的治疗措施（反向因果），已从候选变量中剔除；"
            "机械通气、脓毒症、镇静药暴露均限定于入 ICU 24 小时内。\n")

print("\n[多变量 logistic 结果]")
print(tab3.to_string(index=False))

# ---------------- E. 分层模型 Model 1/2/3 ----------------
M1 = ["age", "female", "sofa_noncns", "gcs_min_24h"]
M2 = M1 + [v for v in ["alcohol_abuse", "epilepsy_hx", "bicarbonate_min",
                       "resp_rate_mean", "temperature_max"] if v in CAND]
M3 = M2 + [v for v in ["sepsis3_24h", "mech_vent_24h", "benzo_24h",
                       "opioid_24h", "steroid_24h"] if v in CAND]
models = {"Model 1\n人口学+器官功能": M1,
          "Model 2\n+合并症与生理": M2,
          "Model 3\n+器官支持与药物": M3}
blocks, aucs_m = {}, {}
for nm, vs in models.items():
    r, a = pool_mice(vs)
    r["label"] = r.variable.map(LABEL)
    blocks[nm] = r.set_index("variable")
    aucs_m[nm] = a

allv = M3
rows_m = []
for v in allv:
    row = {"变量": LABEL[v]}
    for nm in models:
        b = blocks[nm]
        if v in b.index:
            rr = b.loc[v]
            star = "***" if rr.P < 0.001 else ("**" if rr.P < 0.01 else ("*" if rr.P < 0.05 else ""))
            row[nm.replace("\n", " ")] = f"{rr.OR:.2f} ({rr.lo:.2f}-{rr.hi:.2f}){star}"
        else:
            row[nm.replace("\n", " ")] = "—"
    rows_m.append(row)
tab4 = pd.DataFrame(rows_m)
auc_row = {"变量": "模型 AUC"}
for nm in models:
    auc_row[nm.replace("\n", " ")] = f"{aucs_m[nm]:.3f}"
tab4 = pd.concat([tab4, pd.DataFrame([auc_row])], ignore_index=True)
tab4.to_csv(os.path.join(OUT, "table4_stepwise_models.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "table4_stepwise_models.md"), "w", encoding="utf-8") as f:
    f.write("# Table 4. 逐层校正的多变量 logistic 模型（aOR, 95% CI）\n\n")
    f.write(f"主分析集 N={len(main)}，谵妄 {n_ev} 例。多重插补 m={M}，Rubin 规则合并。"
            f"* P<0.05, ** P<0.01, *** P<0.001\n\n")
    f.write(tab4.to_markdown(index=False))
    f.write("\n\n- Model 1：年龄、性别、非神经 SOFA、24h 最低 GCS\n"
            "- Model 2：Model 1 + 酒精滥用、癫痫病史、最低碳酸氢根、呼吸频率、最高体温\n"
            "- Model 3：Model 2 + 脓毒症、有创机械通气、苯二氮卓、阿片类、ICU 24h 激素\n")
print("\n[Table 4 逐层模型]")
print(tab4.to_string(index=False))

# ---------------- 森林图 ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fp = mi_res.sort_values("OR").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(fp) + 2.0), dpi=160)
ypos = np.arange(len(fp))
ax.errorbar(fp.OR, ypos,
            xerr=[fp.OR - fp.lo, fp.hi - fp.OR],
            fmt="o", color="#185FA5", ecolor="#85B7EB",
            elinewidth=1.6, capsize=3, markersize=5)
ax.axvline(1, color="#888780", lw=0.8, ls="--")
ax.set_yticks(ypos)
ax.set_yticklabels(fp.label, fontsize=10)
ax.set_xscale("log")
ax.set_xlabel("调整后比值比 aOR (95% CI，对数刻度)", fontsize=11)
ax.set_title(f"ICU 谵妄独立危险因素（SLE 重症患者，N={len(main)}，事件={n_ev}）", fontsize=12)
for i, r in fp.iterrows():
    ax.text(1.02, i, f"  {r.OR:.2f} ({r.lo:.2f}-{r.hi:.2f})",
            transform=ax.get_yaxis_transform(), va="center", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(max(0.05, fp.lo.min() * 0.7), fp.hi.max() * 1.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_forest.png"), bbox_inches="tight")
print(f"[森林图] 已保存 output/fig_forest.png")

# 供后续脚本使用
main.to_csv(os.path.join(ROOT, "data", "main_analysis_set.csv"), index=False, encoding="utf-8-sig")
pd.Series(FINAL).to_csv(os.path.join(OUT, "final_model_vars.csv"), index=False, header=["variable"])
print("\n[完成] 主分析集已导出 data/main_analysis_set.csv")
