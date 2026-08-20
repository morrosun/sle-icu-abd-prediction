-- ============================================================
-- 20_eicu_cohort.sql   eICU 外部验证队列构建
-- 对齐 MIMIC 主模型的 10 个预测变量：
--   age, female, sofa_noncns, gcs_min_24h, bicarbonate_min,
--   sepsis3_24h, mech_vent_24h, benzo_24h, opioid_24h, steroid_24h
-- 结局（已确认口径）：诊断编码(谵妄/脑病/意识改变/昏迷) 或 GCS<=12
-- ============================================================
\timing on
set search_path to sleci, eicu_crd, public;

-- ---------- 1. SLE 队列（诊断 + 入院用药双重识别） ----------
drop table if exists sleci.ecohort cascade;
create table sleci.ecohort as
with sle_ids as (
  select distinct patientunitstayid from sleci.cohort_raw
),
p as (
  select pt.patientunitstayid          as stay_id,
         pt.patienthealthsystemstayid  as hadm_id,
         pt.uniquepid,
         pt.hospitalid,
         pt.unitvisitnumber,
         case when pt.age = '> 89' then 91.0
              when pt.age ~ '^[0-9]+$' then pt.age::numeric
              else null end            as age,
         case when lower(pt.gender) = 'female' then 1
              when lower(pt.gender) = 'male'   then 0
              else null end            as female,
         pt.ethnicity,
         pt.unitdischargeoffset,
         pt.unitdischargeoffset / 1440.0 as los_icu,
         case when lower(pt.hospitaldischargestatus) = 'expired' then 1
              when lower(pt.hospitaldischargestatus) = 'alive'   then 0
              else null end            as hospital_expire_flag,
         case when lower(pt.unitdischargestatus) = 'expired' then 1 else 0 end as icu_expire_flag,
         pt.apacheadmissiondx,
         pt.admissionweight, pt.admissionheight
  from eicu_crd.patient pt
  join sle_ids s on s.patientunitstayid = pt.patientunitstayid
),
-- 每位患者取首次 ICU 入住（与 MIMIC 口径一致）
first_stay as (
  select p.*,
         row_number() over (partition by uniquepid order by unitvisitnumber, stay_id) as rn
  from p
)
select stay_id, hadm_id, uniquepid, hospitalid, age, female, ethnicity,
       los_icu, hospital_expire_flag, icu_expire_flag,
       apacheadmissiondx, admissionweight, admissionheight
from first_stay
where rn = 1
  and age >= 18
  and unitdischargeoffset >= 1440;   -- ICU LOS >= 24h

create unique index on sleci.ecohort (stay_id);
analyze sleci.ecohort;
select count(*) as eicu_cohort_n, count(distinct uniquepid) as n_patients from sleci.ecohort;

-- ---------- 2. 排除原发急性脑损伤（与 MIMIC 主分析一致） ----------
drop table if exists sleci.eexcl cascade;
create table sleci.eexcl as
select c.stay_id,
  max(case when lower(dx.diagnosisstring) ~ 'traumatic brain|head trauma|skull fracture|cerebral contusion' then 1 else 0 end) as tbi,
  max(case when lower(dx.diagnosisstring) ~ 'intracerebral hemorrhage|intracranial hemorrhage|subarachnoid hemorrhage|subdural|epidural hematoma' then 1 else 0 end) as ich,
  max(case when lower(dx.diagnosisstring) ~ 'ischemic stroke|cerebral infarct|cerebrovascular accident|CVA' then 1 else 0 end) as ischemic_stroke,
  max(case when lower(dx.diagnosisstring) ~ 'meningitis|encephalitis|brain abscess|cerebral abscess' then 1 else 0 end) as cns_infection,
  max(case when lower(dx.diagnosisstring) ~ 'brain tumor|brain metastas|glioma|cerebral neoplasm' then 1 else 0 end) as brain_tumor
from sleci.ecohort c
left join eicu_crd.diagnosis dx on dx.patientunitstayid = c.stay_id
group by c.stay_id;
create unique index on sleci.eexcl (stay_id);

select count(*) n,
       sum(greatest(tbi,ich,ischemic_stroke,cns_infection,brain_tumor)) as any_primary_brain_injury
from sleci.eexcl;

-- ---------- 3. GCS（入 ICU 24h 内最低值） ----------
drop table if exists sleci.egcs cascade;
create table sleci.egcs as
with nc as (
  select n.patientunitstayid as stay_id,
         n.nursingchartoffset as off,
         n.nursingchartcelltypevalname as nm,
         case when n.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
              then n.nursingchartvalue::numeric else null end as val
  from eicu_crd.nursecharting n
  join sleci.ecohort c on c.stay_id = n.patientunitstayid
  where n.nursingchartcelltypevallabel in ('Glasgow coma score','Score (Glasgow Coma Scale)')
    and n.nursingchartoffset between -60 and 1440
),
tot as (
  select stay_id, off, val
  from nc
  where nm in ('GCS Total','Value') and val between 3 and 15
),
comp as (   -- 无 GCS Total 时由 Eyes+Motor+Verbal 求和
  select stay_id, off,
         sum(case when nm='Eyes'   then val end) e,
         sum(case when nm='Motor'  then val end) m,
         sum(case when nm='Verbal' then val end) v
  from nc where nm in ('Eyes','Motor','Verbal')
  group by stay_id, off
),
comp2 as (
  select stay_id, off, (e+m+v) as val
  from comp where e is not null and m is not null and v is not null
    and (e+m+v) between 3 and 15
),
allg as (select * from tot union all select * from comp2)
select stay_id,
       min(val)::numeric  as gcs_min_24h,
       max(val)::numeric  as gcs_max_24h,
       count(*)           as n_gcs
from allg group by stay_id;
create unique index on sleci.egcs (stay_id);
select count(*) with_gcs, round(avg(gcs_min_24h),1) mean_gcsmin from sleci.egcs;

-- ---------- 4. APACHE 生理变量（含 GCS 分量、通气、透析） ----------
drop table if exists sleci.eaps cascade;
create table sleci.eaps as
select c.stay_id,
       a.vent, a.intubated, a.dialysis,
       nullif(a.eyes,-1) e, nullif(a.motor,-1) m, nullif(a.verbal,-1) v,
       nullif(a.urine,-1)        as urine_24h,
       nullif(a.wbc,-1)          as wbc,
       nullif(a.temperature,-1)  as temperature,
       nullif(a.respiratoryrate,-1) as resp_rate,
       nullif(a.sodium,-1)       as sodium,
       nullif(a.heartrate,-1)    as heart_rate,
       nullif(a.meanbp,-1)       as mbp,
       nullif(a.ph,-1)           as ph,
       nullif(a.creatinine,-1)   as creatinine,
       nullif(a.albumin,-1)      as albumin,
       nullif(a.pao2,-1)         as pao2,
       nullif(a.fio2,-1)         as fio2,
       nullif(a.bun,-1)          as bun,
       nullif(a.glucose,-1)      as glucose,
       nullif(a.bilirubin,-1)    as bilirubin,
       apr.apachescore
from sleci.ecohort c
left join eicu_crd.apacheapsvar a on a.patientunitstayid = c.stay_id
left join (
  select patientunitstayid, max(apachescore) apachescore
  from eicu_crd.apachepatientresult
  where apachescore > 0 group by 1
) apr on apr.patientunitstayid = c.stay_id;
create unique index on sleci.eaps (stay_id);

-- ---------- 5. 实验室（入 ICU 24h 内） ----------
drop table if exists sleci.elab cascade;
create table sleci.elab as
select l.patientunitstayid as stay_id,
  min(case when l.labname in ('bicarbonate','HCO3') then l.labresult end)  as bicarbonate_min,
  max(case when l.labname = 'WBC x 1000'        then l.labresult end)      as wbc_max,
  min(case when l.labname = 'platelets x 1000'  then l.labresult end)      as platelets_min,
  min(case when l.labname = 'Hgb'               then l.labresult end)      as hemoglobin_min,
  max(case when l.labname = 'creatinine'        then l.labresult end)      as creatinine_max,
  max(case when l.labname = 'BUN'               then l.labresult end)      as bun_max,
  min(case when l.labname = 'albumin'           then l.labresult end)      as albumin_min,
  max(case when l.labname = 'total bilirubin'   then l.labresult end)      as bilirubin_max,
  max(case when l.labname = 'lactate'           then l.labresult end)      as lactate_max,
  max(case when l.labname = 'glucose'           then l.labresult end)      as glucose_max,
  max(case when l.labname = 'PT - INR'          then l.labresult end)      as inr_max,
  min(case when l.labname = 'sodium'            then l.labresult end)      as sodium_min,
  max(case when l.labname = 'anion gap'         then l.labresult end)      as aniongap_max
from eicu_crd.lab l
join sleci.ecohort c on c.stay_id = l.patientunitstayid
where l.labresultoffset between -360 and 1440
group by l.patientunitstayid;
create unique index on sleci.elab (stay_id);

-- ---------- 6. 机械通气（24h 内有创） ----------
drop table if exists sleci.event cascade;
create table sleci.event as
select c.stay_id,
  greatest(
    coalesce(max(case when lower(t.treatmentstring) ~ 'mechanical ventilation|intubation|ventilation.*assist controlled|ventilation.*pressure controlled|tracheostomy'
                   and t.treatmentoffset <= 1440 then 1 else 0 end),0),
    coalesce(max(case when rc.ventstartoffset <= 1440 and rc.ventstartoffset is not null then 1 else 0 end),0),
    coalesce(max(case when a.vent = 1 or a.intubated = 1 then 1 else 0 end),0)
  ) as mech_vent_24h
from sleci.ecohort c
left join eicu_crd.treatment t       on t.patientunitstayid = c.stay_id
left join eicu_crd.respiratorycare rc on rc.patientunitstayid = c.stay_id
left join sleci.eaps a                on a.stay_id = c.stay_id
group by c.stay_id;
create unique index on sleci.event (stay_id);

-- ---------- 7. 药物暴露（24h 内：苯二氮卓 / 阿片 / 激素） ----------
drop table if exists sleci.edrug cascade;
create table sleci.edrug as
with med as (   -- 处方表
  select m.patientunitstayid as stay_id, lower(m.drugname) as dn, m.drugstartoffset as off
  from eicu_crd.medication m
  join sleci.ecohort c on c.stay_id = m.patientunitstayid
  where m.drugstartoffset between -360 and 1440
),
inf as (        -- 输注表
  select i.patientunitstayid as stay_id, lower(i.drugname) as dn, i.infusionoffset as off
  from eicu_crd.infusiondrug i
  join sleci.ecohort c on c.stay_id = i.patientunitstayid
  where i.infusionoffset between -360 and 1440
),
allrx as (select * from med union all select * from inf)
select c.stay_id,
  coalesce(max(case when a.dn ~ 'midazolam|lorazepam|diazepam|versed|ativan|clonazepam|alprazolam|chlordiazepoxide'
                    then 1 else 0 end),0) as benzo_24h,
  coalesce(max(case when a.dn ~ 'fentanyl|morphine|hydromorphone|dilaudid|oxycodone|remifentanil|sufentanil|meperidine|methadone|tramadol|codeine|buprenorphine'
                    then 1 else 0 end),0) as opioid_24h,
  coalesce(max(case when a.dn ~ 'propofol|diprivan' then 1 else 0 end),0) as propofol_24h,
  coalesce(max(case when a.dn ~ 'dexmedetomidine|precedex' then 1 else 0 end),0) as dexmed_24h,
  coalesce(max(case when a.dn ~ 'haloperidol|haldol' then 1 else 0 end),0) as haloperidol,
  coalesce(max(case when a.dn ~ 'quetiapine|olanzapine|risperidone|ziprasidone|aripiprazole'
                    then 1 else 0 end),0) as atypical_ap,
  -- 全身性糖皮质激素（排除外用/吸入/眼耳制剂）
  coalesce(max(case when a.dn ~ 'methylprednisolone|solu-medrol|prednisone|prednisolone|dexamethasone|decadron|hydrocortisone|solu-cortef|cortisone|betamethasone'
                and a.dn !~ 'cream|ointment|topical|ophthalmic|otic|inhal|nasal|neomycin|cortisporin'
                    then 1 else 0 end),0) as steroid_24h,
  coalesce(max(case when a.dn ~ 'hydroxychloroquine|plaquenil' then 1 else 0 end),0) as hcq,
  coalesce(max(case when a.dn ~ 'mycophenolate|cellcept|myfortic' then 1 else 0 end),0) as mmf,
  coalesce(max(case when a.dn ~ 'azathioprine|imuran' then 1 else 0 end),0) as aza,
  coalesce(max(case when a.dn ~ 'cyclophosphamide|cytoxan' then 1 else 0 end),0) as cyc,
  coalesce(max(case when a.dn ~ 'tacrolimus|cyclosporine|prograf|neoral' then 1 else 0 end),0) as cni,
  coalesce(max(case when a.dn ~ 'norepinephrine|epinephrine|dopamine|vasopressin|phenylephrine|levophed|dobutamine'
                    then 1 else 0 end),0) as vaso_24h
from sleci.ecohort c
left join allrx a on a.stay_id = c.stay_id
group by c.stay_id;
create unique index on sleci.edrug (stay_id);

-- ---------- 8. 脓毒症（诊断字符串 + APACHE 入院诊断） ----------
drop table if exists sleci.esepsis cascade;
create table sleci.esepsis as
select c.stay_id,
  greatest(
    coalesce(max(case when lower(dx.diagnosisstring) ~ 'sepsis|septic shock|septicemia'
                   and dx.diagnosisoffset <= 1440 then 1 else 0 end),0),
    case when lower(c.apacheadmissiondx) ~ 'sepsis|septic' then 1 else 0 end
  ) as sepsis3_24h
from sleci.ecohort c
left join eicu_crd.diagnosis dx on dx.patientunitstayid = c.stay_id
group by c.stay_id, c.apacheadmissiondx;
create unique index on sleci.esepsis (stay_id);

-- ---------- 9. 合并症（pasthistory + diagnosis） ----------
drop table if exists sleci.ecomorbid cascade;
create table sleci.ecomorbid as
with ph as (
  select p.patientunitstayid as stay_id, lower(p.pasthistorypath || ' ' || coalesce(p.pasthistoryvalue,'')) as s
  from eicu_crd.pasthistory p join sleci.ecohort c on c.stay_id = p.patientunitstayid
),
dx as (
  select d.patientunitstayid as stay_id, lower(d.diagnosisstring) as s
  from eicu_crd.diagnosis d join sleci.ecohort c on c.stay_id = d.patientunitstayid
),
allh as (select * from ph union all select * from dx)
select c.stay_id,
  coalesce(max(case when a.s ~ 'hypertension' then 1 else 0 end),0)                    as htn,
  coalesce(max(case when a.s ~ 'diabetes' then 1 else 0 end),0)                        as diabetes,
  coalesce(max(case when a.s ~ 'renal insufficiency|chronic kidney|renal failure' then 1 else 0 end),0) as ckd,
  coalesce(max(case when a.s ~ 'dialysis|esrd|end-stage renal' then 1 else 0 end),0)   as esrd_dialysis,
  coalesce(max(case when a.s ~ 'seizure|epilep' then 1 else 0 end),0)                  as epilepsy_hx,
  coalesce(max(case when a.s ~ 'dementia|alzheimer' then 1 else 0 end),0)              as dementia_hx,
  coalesce(max(case when a.s ~ 'depress' then 1 else 0 end),0)                         as depression_hx,
  coalesce(max(case when a.s ~ 'alcohol' then 1 else 0 end),0)                         as alcohol_abuse,
  coalesce(max(case when a.s ~ 'antiphospholipid|lupus anticoagulant' then 1 else 0 end),0) as aps_antiphospholipid,
  coalesce(max(case when a.s ~ 'heart failure|cardiomyopathy|chf' then 1 else 0 end),0) as chf,
  coalesce(max(case when a.s ~ 'copd|emphysema|chronic bronchitis' then 1 else 0 end),0) as copd,
  coalesce(max(case when a.s ~ 'cancer|malignan|carcinoma|lymphoma|leukemia' then 1 else 0 end),0) as malignancy,
  coalesce(max(case when a.s ~ 'lupus nephritis|nephritis.*lupus' then 1 else 0 end),0) as lupus_nephritis_icd
from sleci.ecohort c
left join allh a on a.stay_id = c.stay_id
group by c.stay_id;
create unique index on sleci.ecomorbid (stay_id);

-- ---------- 10. 结局：急性脑功能障碍（诊断编码 + GCS 口径） ----------
drop table if exists sleci.eoutcome cascade;
create table sleci.eoutcome as
with dxout as (
  select c.stay_id,
    max(case when lower(dx.diagnosisstring) ~ 'delirium|acute confusion' then 1 else 0 end) as dx_delirium,
    max(case when lower(dx.diagnosisstring) ~ 'encephalopath' then 1 else 0 end)            as dx_encephalopathy,
    max(case when lower(dx.diagnosisstring) ~ 'change in mental status|altered mental status|obtundation|unresponsive' then 1 else 0 end) as dx_ams,
    max(case when lower(dx.diagnosisstring) ~ 'coma' then 1 else 0 end)                     as dx_coma,
    max(case when lower(dx.diagnosisstring) ~ 'seizure|status epilepticus' then 1 else 0 end) as dx_seizure,
    max(case when lower(dx.diagnosisstring) ~ 'psychosis|psychotic' then 1 else 0 end)      as dx_psychosis
  from sleci.ecohort c
  left join eicu_crd.diagnosis dx on dx.patientunitstayid = c.stay_id
  group by c.stay_id
),
ncdel as (   -- 护理记录中的谵妄量表（覆盖率低，仅作参考）
  select n.patientunitstayid as stay_id,
    max(case when upper(n.nursingchartvalue) = 'YES' then 1 else 0 end) as nc_delirium_pos,
    1 as nc_assessed
  from eicu_crd.nursecharting n
  join sleci.ecohort c on c.stay_id = n.patientunitstayid
  where n.nursingchartcelltypevallabel in ('Delirium Scale/Score','Symptoms of Delirium Present')
  group by n.patientunitstayid
)
select c.stay_id,
       coalesce(d.dx_delirium,0)        as dx_delirium,
       coalesce(d.dx_encephalopathy,0)  as dx_encephalopathy,
       coalesce(d.dx_ams,0)             as dx_ams,
       coalesce(d.dx_coma,0)            as dx_coma,
       coalesce(d.dx_seizure,0)         as dx_seizure,
       coalesce(d.dx_psychosis,0)       as dx_psychosis,
       coalesce(nc.nc_delirium_pos,0)   as nc_delirium_pos,
       coalesce(nc.nc_assessed,0)       as nc_assessed,
       g.gcs_min_24h,
       -- 主验证结局：诊断编码任一阳性 或 24h 内 GCS<=12
       case when coalesce(d.dx_delirium,0) = 1
              or coalesce(d.dx_encephalopathy,0) = 1
              or coalesce(d.dx_ams,0) = 1
              or coalesce(d.dx_coma,0) = 1
              or coalesce(nc.nc_delirium_pos,0) = 1
              or (g.gcs_min_24h is not null and g.gcs_min_24h <= 12)
            then 1 else 0 end as abd_dx_gcs,
       -- 严格口径（仅诊断编码，敏感性分析）
       case when coalesce(d.dx_delirium,0) = 1
              or coalesce(d.dx_encephalopathy,0) = 1
              or coalesce(d.dx_coma,0) = 1
            then 1 else 0 end as abd_dx_only
from sleci.ecohort c
left join dxout d  on d.stay_id = c.stay_id
left join ncdel nc on nc.stay_id = c.stay_id
left join sleci.egcs g on g.stay_id = c.stay_id;
create unique index on sleci.eoutcome (stay_id);

select count(*) n,
       sum(dx_delirium) dlr, sum(dx_encephalopathy) enc, sum(dx_ams) ams,
       sum(dx_coma) coma, sum(nc_delirium_pos) ncpos,
       sum(case when gcs_min_24h <= 12 then 1 else 0 end) gcs_le12,
       sum(abd_dx_gcs) composite, sum(abd_dx_only) dx_only
from sleci.eoutcome;
