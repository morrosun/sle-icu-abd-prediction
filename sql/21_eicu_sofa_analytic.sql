-- ============================================================
-- 21_eicu_sofa_analytic.sql
-- eICU 非神经 SOFA 计算 + 外部验证分析宽表
-- SOFA 五个非神经系统分量（呼吸/凝血/肝/心血管/肾），与 MIMIC sofa_noncns 对齐
-- ============================================================
\timing on
set search_path to sleci, eicu_crd, public;

drop table if exists sleci.esofa cascade;
create table sleci.esofa as
with base as (
  select c.stay_id,
         a.pao2, a.fio2, a.vent, a.intubated, a.mbp, a.urine_24h,
         l.platelets_min, l.bilirubin_max, l.creatinine_max,
         d.vaso_24h
  from sleci.ecohort c
  left join sleci.eaps  a on a.stay_id = c.stay_id
  left join sleci.elab  l on l.stay_id = c.stay_id
  left join sleci.edrug d on d.stay_id = c.stay_id
),
calc as (
  select stay_id,
    -- FiO2 归一化到 0-1
    case when fio2 is null then null
         when fio2 > 1 then fio2/100.0
         else fio2 end as fio2n,
    pao2, vent, intubated, mbp, urine_24h,
    platelets_min, bilirubin_max, creatinine_max, vaso_24h
  from base
),
pf as (
  select *, case when pao2 is not null and fio2n is not null and fio2n > 0
                 then pao2 / fio2n else null end as pfratio
  from calc
)
select stay_id,
  -- 呼吸
  case when pfratio is null then 0
       when pfratio < 100 and coalesce(vent,intubated,0) = 1 then 4
       when pfratio < 200 and coalesce(vent,intubated,0) = 1 then 3
       when pfratio < 300 then 2
       when pfratio < 400 then 1
       else 0 end as sofa_resp,
  -- 凝血
  case when platelets_min is null then 0
       when platelets_min < 20  then 4
       when platelets_min < 50  then 3
       when platelets_min < 100 then 2
       when platelets_min < 150 then 1
       else 0 end as sofa_coag,
  -- 肝
  case when bilirubin_max is null then 0
       when bilirubin_max >= 12.0 then 4
       when bilirubin_max >= 6.0  then 3
       when bilirubin_max >= 2.0  then 2
       when bilirubin_max >= 1.2  then 1
       else 0 end as sofa_liver,
  -- 心血管（eICU 无剂量分级，用血管活性药 + MAP 近似）
  case when coalesce(vaso_24h,0) = 1 then 3
       when mbp is not null and mbp < 70 then 1
       else 0 end as sofa_cv,
  -- 肾
  case when creatinine_max is not null and creatinine_max >= 5.0 then 4
       when urine_24h is not null and urine_24h < 200 then 4
       when creatinine_max is not null and creatinine_max >= 3.5 then 3
       when urine_24h is not null and urine_24h < 500 then 3
       when creatinine_max is not null and creatinine_max >= 2.0 then 2
       when creatinine_max is not null and creatinine_max >= 1.2 then 1
       else 0 end as sofa_renal,
  pfratio
from pf;

alter table sleci.esofa add column sofa_noncns int;
update sleci.esofa
set sofa_noncns = sofa_resp + sofa_coag + sofa_liver + sofa_cv + sofa_renal;
create unique index on sleci.esofa (stay_id);

select count(*) n,
       round(avg(sofa_noncns),2) mean_sofa,
       percentile_cont(0.5) within group (order by sofa_noncns) median_sofa,
       min(sofa_noncns) mn, max(sofa_noncns) mx
from sleci.esofa;

-- ---------- 分析宽表 ----------
drop table if exists sleci.eanalytic cascade;
create table sleci.eanalytic as
select
  c.stay_id, c.uniquepid, c.hospitalid,
  c.age, c.female, c.ethnicity, c.los_icu,
  c.hospital_expire_flag, c.icu_expire_flag,
  -- 排除标志
  greatest(x.tbi, x.ich, x.ischemic_stroke, x.cns_infection, x.brain_tumor) as primary_brain_injury,
  x.tbi, x.ich, x.ischemic_stroke, x.cns_infection, x.brain_tumor,
  -- 模型预测变量（与 MIMIC 对齐）
  s.sofa_noncns, s.sofa_resp, s.sofa_coag, s.sofa_liver, s.sofa_cv, s.sofa_renal,
  g.gcs_min_24h, g.n_gcs,
  l.bicarbonate_min, l.wbc_max, l.platelets_min, l.hemoglobin_min,
  l.creatinine_max, l.bun_max, l.albumin_min, l.lactate_max,
  l.glucose_max, l.inr_max, l.sodium_min, l.aniongap_max,
  sp.sepsis3_24h,
  v.mech_vent_24h,
  dr.benzo_24h, dr.opioid_24h, dr.propofol_24h, dr.dexmed_24h,
  dr.steroid_24h, dr.hcq, dr.mmf, dr.aza, dr.cyc, dr.cni, dr.vaso_24h,
  dr.haloperidol, dr.atypical_ap,
  a.temperature as temperature_max, a.resp_rate as resp_rate_mean,
  a.heart_rate as heart_rate_mean, a.mbp as mbp_min, a.urine_24h as uo_24h,
  a.apachescore,
  -- 合并症
  cm.htn, cm.diabetes, cm.ckd, cm.esrd_dialysis, cm.epilepsy_hx, cm.dementia_hx,
  cm.depression_hx, cm.alcohol_abuse, cm.aps_antiphospholipid, cm.chf, cm.copd,
  cm.malignancy, cm.lupus_nephritis_icd,
  -- 结局
  o.abd_dx_gcs, o.abd_dx_only,
  o.dx_delirium, o.dx_encephalopathy, o.dx_ams, o.dx_coma, o.dx_seizure, o.dx_psychosis,
  o.nc_delirium_pos, o.nc_assessed
from sleci.ecohort c
left join sleci.eexcl     x  on x.stay_id  = c.stay_id
left join sleci.esofa     s  on s.stay_id  = c.stay_id
left join sleci.egcs      g  on g.stay_id  = c.stay_id
left join sleci.elab      l  on l.stay_id  = c.stay_id
left join sleci.esepsis   sp on sp.stay_id = c.stay_id
left join sleci.event     v  on v.stay_id  = c.stay_id
left join sleci.edrug     dr on dr.stay_id = c.stay_id
left join sleci.eaps      a  on a.stay_id  = c.stay_id
left join sleci.ecomorbid cm on cm.stay_id = c.stay_id
left join sleci.eoutcome  o  on o.stay_id  = c.stay_id;

create unique index on sleci.eanalytic (stay_id);
analyze sleci.eanalytic;

-- 验证集摘要（排除原发脑损伤，且有 GCS 记录）
select count(*) n_all,
       sum(case when primary_brain_injury = 0 then 1 else 0 end) after_excl,
       sum(case when primary_brain_injury = 0 and gcs_min_24h is not null then 1 else 0 end) with_gcs,
       sum(case when primary_brain_injury = 0 and gcs_min_24h is not null then abd_dx_gcs else 0 end) events
from sleci.eanalytic;
