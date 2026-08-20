-- ============================================================
-- 24_nwicu_transfer.sql
-- NWICU 模型可迁移性 / 阳性对照 分析数据构建
--
-- 治疗组 (enc_group=1) : nwicu_sae.cohort 的 165 例脑病 ICU 队列
-- 对照组 (enc_group=0) : 首次入住 / 年龄>=18 / LOS>=24h / 非脑病 / 非原发脑损伤 的随机 600 例
--
-- 特征对齐 MIMIC 无 GCS 模型口径（见 code/04 FEATS 去 gcs_min_24h）：
--   age, female, sofa_noncns(近似), sepsis3_24h, mech_vent_24h,
--   benzo/opioid/steroid/propofol/dexmed/vaso/hcq,
--   bicarbonate_min, wbc_max, platelets_min, hemoglobin_min,
--   creatinine_max, bun_max, lactate_max, glucose_max, sodium_min,
--   bilirubin_max, inr_max, temperature_max(F->C), heart_rate_mean,
--   resp_rate_mean, mbp_min, htn, diabetes, ckd, chf, malignancy,
--   epilepsy_hx, lupus_nephritis_icd
-- 注: 体温 NWICU 为华氏度，已换算摄氏; sofa_noncns 由分项近似汇总
-- ============================================================
\timing on
set search_path to nwicu_sae, icu, hosp, public;

-- 0. 治疗组 = 165 脑病队列; 对照组池 = 首次入住/年龄>=18/LOS>=24h/非原发脑损伤
drop table if exists nwicu_sae.transfer cascade;
create table nwicu_sae.transfer as
with enc_stays as (
  select stay_id, hadm_id, subject_id from nwicu_sae.cohort
),
fs as (
  select s.stay_id, s.subject_id, s.hadm_id, s.intime, s.outtime,
         p.anchor_age as age, p.gender,
         row_number() over (partition by s.subject_id order by s.intime) rn
  from icu.icustays s
  join hosp.patients p on p.subject_id = s.subject_id
),
fs1 as (select * from fs where rn = 1 and age >= 18
        and extract(epoch from (outtime-intime))/86400.0 >= 1.0),
label as (
  select fs1.stay_id,
    exists (select 1 from hosp.diagnoses_icd d
            where d.subject_id=fs1.subject_id and d.hadm_id=fs1.hadm_id
              and d.icd_code ~* '^(F05|G92|G93.4|R40.2|R40.3|R40.4|R41.0|R41.82|R41.89)') as enc,
    exists (select 1 from hosp.diagnoses_icd d
            where d.subject_id=fs1.subject_id and d.hadm_id=fs1.hadm_id
              and d.icd_code ~* '^(S06|I61|I62|I63|I64|G45|G00|G04|G07|A80|A81|A83|A84|A85|A86|B00.4|C70|C71|C72|C79.3)') as pbi
  from fs1
),
pool as (
  select fs1.stay_id, fs1.subject_id, fs1.hadm_id, fs1.intime, fs1.age, fs1.gender, l.enc
  from fs1 join label l using(stay_id) where not l.pbi
),
ctrl as (
  select stay_id, subject_id, hadm_id, intime, age, gender
  from pool
  where not enc and stay_id not in (select stay_id from enc_stays)
  order by random() limit 600
),
trt as (
  select e.stay_id, s.subject_id, s.hadm_id, s.intime,
         p.anchor_age as age, p.gender
  from enc_stays e
  join icu.icustays s on s.stay_id = e.stay_id
  join hosp.patients p on p.subject_id = e.subject_id
),
universe as (
  select stay_id, subject_id, hadm_id, intime, age, gender from trt
  union all
  select stay_id, subject_id, hadm_id, intime, age, gender from ctrl
),
-- 实验室 (入 ICU 24h 内, 含 -24h 取材)
labs as (
  select u.stay_id,
    max(case when li.itemid = 100031 and le.valuenum between 0.1 and 50 then le.valuenum end) as lactate_max,
    min(case when li.label ~* 'bicarb|hco3|co2|carbon dioxide' and le.valuenum between 1 and 80 then le.valuenum end) as bicarbonate_min,
    max(case when li.itemid in (100083,100016) and le.valuenum between 0.01 and 500 then le.valuenum end) as wbc_max,
    min(case when li.itemid = 100014 and le.valuenum between 1 and 2000 then le.valuenum end) as platelets_min,
    min(case when li.itemid = 100007 and le.valuenum between 1 and 30 then le.valuenum end) as hemoglobin_min,
    max(case when li.itemid = 100002 and le.valuenum between 0.1 and 30 then le.valuenum end) as creatinine_max,
    max(case when li.itemid = 100004 and le.valuenum between 1 and 400 then le.valuenum end) as bun_max,
    max(case when li.itemid in (100001,100045,100062) and le.valuenum between 5 and 2000 then le.valuenum end) as glucose_max,
    min(case when li.itemid = 100010 and le.valuenum between 80 and 200 then le.valuenum end) as sodium_min,
    max(case when li.itemid = 100020 and le.valuenum between 0.1 and 100 then le.valuenum end) as bilirubin_max,
    max(case when li.itemid = 100030 and le.valuenum between 0.5 and 20 then le.valuenum end) as inr_max,
    max(case when li.itemid in (100041,100040) and le.valuenum between 1 and 60 then le.valuenum end) as aniongap_max,
    min(case when li.itemid = 100021 and le.valuenum between 0.1 and 10 then le.valuenum end) as albumin_min,
    min(case when li.itemid = 100003 and le.valuenum between 1 and 30 then le.valuenum end) as calcium_min,
    max(case when li.itemid = 100008 and le.valuenum between 5 and 40 then le.valuenum end) as rdw_max
  from universe u
  join hosp.labevents le on le.subject_id = u.subject_id and le.hadm_id = u.hadm_id
  join hosp.d_labitems li on li.itemid = le.itemid
  where le.charttime between u.intime - interval '24 hours' and u.intime + interval '24 hours'
    and le.valuenum is not null
  group by u.stay_id
),
-- 药物 (入 ICU 24h 窗口内)
meds as (
  select u.stay_id,
    max(case when lower(p.drug) ~ 'midazolam|lorazepam|diazepam|clonazepam|alprazolam|chlordiazepoxide|temazepam|triazolam|versed|ativan|klonopin' then 1 else 0 end) as benzo_24h,
    max(case when lower(p.drug) ~ 'fentanyl|morphine|hydromorphone|dilaudid|oxycodone|remifentanil|sufentanil|meperidine|methadone|tramadol|codeine|hydrocodone|buprenorphine|opiate|opioid' then 1 else 0 end) as opioid_24h,
    max(case when lower(p.drug) ~ 'propofol|diprivan' then 1 else 0 end) as propofol_24h,
    max(case when lower(p.drug) ~ 'dexmedetomidine|precedex' then 1 else 0 end) as dexmed_24h,
    max(case when lower(p.drug) ~ 'hydroxychloroquine|plaquenil' then 1 else 0 end) as hcq,
    max(case when lower(p.drug) ~ 'azathioprine|mycophenolate|cellcept|myfortic|cyclophosphamide|cyclosporine|tacrolimus' then 1 else 0 end) as sle_tx,
    max(case when lower(p.drug) ~ 'norepinephrine|epinephrine|dopamine|vasopressin|phenylephrine|dobutamine|levophed|angiotensin' then 1 else 0 end) as vaso_24h,
    max(case when lower(p.drug) ~ 'methylprednisolone|solu-medrol|prednisone|prednisolone|dexamethasone|decadron|hydrocortisone|solu-cortef|cortisone|betamethasone'
              and lower(p.drug) !~ 'cream|ointment|topical|ophthalmic|otic|inhal|nasal|neomycin|cortisporin|opth|eye|ear' then 1 else 0 end) as steroid_24h
  from universe u
  join hosp.prescriptions p on p.subject_id = u.subject_id and p.hadm_id = u.hadm_id
  where (p.starttime between u.intime - interval '24 hours' and u.intime + interval '24 hours'
         or (p.stoptime is not null and p.stoptime between u.intime - interval '24 hours' and u.intime + interval '24 hours'))
  group by u.stay_id
),
-- 生命体征 (入 ICU 24h 内; 体温华氏->摄氏)
vitals as (
  select u.stay_id,
    max(case when ci.itemid = 323761 then (case when ci.unitname='F' then (ce.valuenum-32)*5.0/9 else ce.valuenum end) end) as temperature_max,
    avg(case when ci.itemid = 320045 then ce.valuenum end) as heart_rate_mean,
    avg(case when ci.itemid = 320210 then ce.valuenum end) as resp_rate_mean,
    avg(case when ci.itemid in (320050,320051) then ce.valuenum end) as mbp_min
  from universe u
  left join icu.chartevents ce on ce.subject_id = u.subject_id and ce.hadm_id = u.hadm_id
    and ce.charttime between u.intime and u.intime + interval '24 hours'
    and ce.valuenum is not null
  left join icu.d_items ci on ci.itemid = ce.itemid
    and ci.itemid in (323761,320045,320210,320050,320051)
  group by u.stay_id
),
-- 机械通气 (插管/气管切开 chart; 或处方通气相关)
vent as (
  select u.stay_id,
    greatest(
      coalesce(max(case when ci.itemid in (753461,763437) then 1 else 0 end),0),
      coalesce(max(case when lower(p.drug) ~ 'ventilator|intubation|tracheostomy|bipap|cpap|niv|non-invasive ventilation|mechanical ventilation' then 1 else 0 end),0)
    ) as mech_vent_24h
  from universe u
  left join icu.chartevents ce on ce.subject_id = u.subject_id and ce.hadm_id = u.hadm_id
    and ce.charttime between u.intime and u.intime + interval '24 hours'
  left join icu.d_items ci on ci.itemid = ce.itemid
  left join hosp.prescriptions p on p.subject_id = u.subject_id and p.hadm_id = u.hadm_id
    and (p.starttime between u.intime - interval '24 hours' and u.intime + interval '24 hours'
         or (p.stoptime is not null and p.stoptime between u.intime - interval '24 hours' and u.intime + interval '24 hours'))
  group by u.stay_id
),
-- 脓毒症 (ICD-10 脓毒症/休克; 叠加 nwicu_sae.sepsis 标记)
sepsis as (
  select u.stay_id,
    max(case when d.icd_code ~* '^(A40|A41|R57.2|R65|B37.7|U82|U83|U89|A39)' then 1 else 0 end) as sepsis3_24h
  from universe u
  left join hosp.diagnoses_icd d on d.subject_id = u.subject_id and d.hadm_id = u.hadm_id
  group by u.stay_id
),
-- 合并症 (基于诊断编码)
comorb as (
  select u.stay_id,
    max(case when d.icd_code ~* '^I1[0-5]' then 1 else 0 end) as htn,
    max(case when d.icd_code ~* '^E1[0-3]' then 1 else 0 end) as diabetes,
    max(case when d.icd_code ~* '^N18' then 1 else 0 end) as ckd,
    max(case when d.icd_code ~* '^N18.5|^Z99.2|^Z49|^T82.0|^Z94' then 1 else 0 end) as esrd_dialysis,
    max(case when d.icd_code ~* '^I50' then 1 else 0 end) as chf,
    max(case when d.icd_code ~* '^C[0-9][0-9]|^C7[0-6]|^C8[0-9]|^C9[0-6]' then 1 else 0 end) as malignancy,
    max(case when d.icd_code ~* '^G40|^R56' then 1 else 0 end) as epilepsy_hx,
    max(case when d.icd_code ~* '^M32.1' then 1 else 0 end) as lupus_nephritis_icd
  from universe u
  left join hosp.diagnoses_icd d on d.subject_id = u.subject_id and d.hadm_id = u.hadm_id
  group by u.stay_id
)
select
  u.stay_id, u.subject_id, u.hadm_id,
  case when e.stay_id is not null then 1 else 0 end as enc_group,
  u.age,
  case when u.gender = 'F' then 1 else 0 end as female,
  lb.lactate_max, lb.bicarbonate_min, lb.wbc_max, lb.platelets_min, lb.hemoglobin_min,
  lb.creatinine_max, lb.bun_max, lb.glucose_max, lb.sodium_min, lb.bilirubin_max,
  lb.inr_max, lb.aniongap_max, lb.albumin_min, lb.calcium_min, lb.rdw_max,
  md.benzo_24h, md.opioid_24h, md.propofol_24h, md.dexmed_24h, md.hcq, md.sle_tx, md.vaso_24h, md.steroid_24h,
  vt.temperature_max, vt.heart_rate_mean, vt.resp_rate_mean, vt.mbp_min,
  vn.mech_vent_24h, sp.sepsis3_24h,
  cb.htn, cb.diabetes, cb.ckd, cb.esrd_dialysis, cb.chf, cb.malignancy, cb.epilepsy_hx, cb.lupus_nephritis_icd
from universe u
left join enc_stays e on e.stay_id = u.stay_id
left join labs lb on lb.stay_id = u.stay_id
left join meds md on md.stay_id = u.stay_id
left join vitals vt on vt.stay_id = u.stay_id
left join vent vn on vn.stay_id = u.stay_id
left join sepsis sp on sp.stay_id = u.stay_id
left join comorb cb on cb.stay_id = u.stay_id;

create unique index on nwicu_sae.transfer (stay_id);
analyze nwicu_sae.transfer;
select enc_group, count(*) n,
       round(100.0*avg((age>=18)::int),0) pct_adult,
       round(avg(age),1) mean_age
from nwicu_sae.transfer group by enc_group;

-- 导出 CSV
\copy (select * from nwicu_sae.transfer) to 'data/nwicu_transfer.csv' with csv header
