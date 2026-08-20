-- SLECI 12: covariates (first 24h of ICU)
\timing on
set work_mem='512MB';

drop table if exists sleci.comorbid cascade;
create table sleci.comorbid as
select c.stay_id,
  max(case when d.icd_code like 'I10%' or d.icd_code like 'I11%' or d.icd_code like 'I12%' or d.icd_code like 'I13%'
            or d.icd_code like '401%' or d.icd_code like '402%' or d.icd_code like '403%' or d.icd_code like '404%' then 1 else 0 end) htn,
  max(case when d.icd_code like 'E11%' or d.icd_code like 'E10%' or d.icd_code like '250%' then 1 else 0 end) diabetes,
  max(case when d.icd_code like 'N18%' or d.icd_code like '585%' then 1 else 0 end) ckd,
  max(case when d.icd_code like 'N185%' or d.icd_code like 'N186%' or d.icd_code like 'Z992%' or d.icd_code like 'Z49%'
            or d.icd_code like '5856%' or d.icd_code like 'V4511%' then 1 else 0 end) esrd_dialysis,
  max(case when d.icd_code like 'G40%' or d.icd_code like '345%' then 1 else 0 end) epilepsy_hx,
  max(case when d.icd_code like 'F03%' or d.icd_code like 'G30%' or d.icd_code like '290%' or d.icd_code like '3310%' then 1 else 0 end) dementia_hx,
  max(case when d.icd_code like 'F32%' or d.icd_code like 'F33%' or d.icd_code like '311%' or d.icd_code like '296%' then 1 else 0 end) depression_hx,
  max(case when d.icd_code like 'F10%' or d.icd_code like '303%' or d.icd_code like '2915%' then 1 else 0 end) alcohol_abuse,
  max(case when d.icd_code like 'D68 61%' or d.icd_code like 'D6861%' or d.icd_code like '28981%' then 1 else 0 end) aps_antiphospholipid,
  max(case when d.icd_code like 'I50%' or d.icd_code like '428%' then 1 else 0 end) chf,
  max(case when d.icd_code like 'J44%' or d.icd_code like '496%' then 1 else 0 end) copd,
  max(case when d.icd_code like 'C%' or d.icd_code like '14%' or d.icd_code like '15%' or d.icd_code like '16%'
            or d.icd_code like '17%' or d.icd_code like '18%' or d.icd_code like '19%' or d.icd_code like '20%' then 1 else 0 end) malignancy,
  max(case when d.icd_code like 'D5%' or d.icd_code like 'D6%' or d.icd_code like '280%' or d.icd_code like '281%'
            or d.icd_code like '282%' or d.icd_code like '283%' or d.icd_code like '284%' or d.icd_code like '285%' then 1 else 0 end) anemia_dx,
  max(case when d.icd_code like 'D693%' or d.icd_code like '2873%' then 1 else 0 end) itp
from sleci.cohort_final c
join mimiciv_hosp.diagnoses_icd d on d.hadm_id = c.hadm_id
group by c.stay_id;
create index on sleci.comorbid (stay_id);

drop table if exists sleci.crp_fd cascade;
create table sleci.crp_fd as
select c.stay_id, max(i.crp) crp_max_24h
from sleci.cohort_final c
join mimiciv_derived.inflammation i on i.hadm_id = c.hadm_id
where i.charttime between c.icu_intime - interval '6 hour' and c.icu_intime + interval '24 hour'
group by c.stay_id;
create index on sleci.crp_fd (stay_id);

drop table if exists sleci.covariates cascade;
create table sleci.covariates as
select c.stay_id,
  -- severity
  fs.sofa sofa_24h, fs.respiration sofa_resp, fs.coagulation sofa_coag, fs.liver sofa_liver,
  fs.cardiovascular sofa_cv, fs.cns sofa_cns, fs.renal sofa_renal,
  ap.apsiii, sp.sapsii, oa.oasis,
  ch.charlson_comorbidity_index charlson,
  -- vitals
  v.heart_rate_mean, v.sbp_mean, v.mbp_min, v.resp_rate_mean,
  v.temperature_max, v.temperature_min, v.spo2_min,
  -- labs
  l.wbc_max, l.wbc_min, l.hemoglobin_min, l.platelets_min, l.albumin_min,
  l.creatinine_max, l.bun_max, l.sodium_min, l.sodium_max, l.potassium_max,
  l.bicarbonate_min, l.aniongap_max, l.glucose_max, l.inr_max, l.ptt_max,
  l.bilirubin_total_max, l.alt_max, l.ast_max, l.calcium_min,
  l.abs_lymphocytes_min, l.abs_neutrophils_max,
  bg.lactate_max, bg.ph_min, bg.pao2fio2ratio_min,
  crp.crp_max_24h,
  -- anthropometrics
  w.weight_admit, h.height,
  case when h.height is not null and h.height>0 then w.weight_admit/((h.height/100.0)^2) end bmi,
  -- urine
  uo.urineoutput uo_24h
from sleci.cohort_final c
left join mimiciv_derived.first_day_sofa      fs using (stay_id)
left join mimiciv_derived.apsiii              ap using (stay_id)
left join mimiciv_derived.sapsii              sp using (stay_id)
left join mimiciv_derived.oasis               oa using (stay_id)
left join mimiciv_derived.charlson            ch on ch.hadm_id = c.hadm_id
left join mimiciv_derived.first_day_vitalsign v using (stay_id)
left join mimiciv_derived.first_day_lab       l using (stay_id)
left join mimiciv_derived.first_day_bg        bg using (stay_id)
left join sleci.crp_fd                        crp using (stay_id)
left join mimiciv_derived.first_day_weight    w using (stay_id)
left join mimiciv_derived.first_day_height    h using (stay_id)
left join mimiciv_derived.first_day_urine_output uo using (stay_id);
create index on sleci.covariates (stay_id);
analyze sleci.covariates;

-- missingness report
select 'sofa' v, 100.0*sum(case when sofa_24h is null then 1 else 0 end)/count(*) pct_miss from sleci.covariates
union all select 'apsiii', 100.0*sum(case when apsiii is null then 1 else 0 end)/count(*) from sleci.covariates
union all select 'albumin', 100.0*sum(case when albumin_min is null then 1 else 0 end)/count(*) from sleci.covariates
union all select 'lactate', 100.0*sum(case when lactate_max is null then 1 else 0 end)/count(*) from sleci.covariates
union all select 'crp', 100.0*sum(case when crp_max_24h is null then 1 else 0 end)/count(*) from sleci.covariates
union all select 'bmi', 100.0*sum(case when bmi is null then 1 else 0 end)/count(*) from sleci.covariates
union all select 'inr', 100.0*sum(case when inr_max is null then 1 else 0 end)/count(*) from sleci.covariates
union all select 'pf_ratio', 100.0*sum(case when pao2fio2ratio_min is null then 1 else 0 end)/count(*) from sleci.covariates
union all select 'lymphocytes', 100.0*sum(case when abs_lymphocytes_min is null then 1 else 0 end)/count(*) from sleci.covariates
union all select 'urine_output', 100.0*sum(case when uo_24h is null then 1 else 0 end)/count(*) from sleci.covariates
order by 2 desc;
