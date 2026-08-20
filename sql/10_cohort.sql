-- SLECI 10: MIMIC-IV analytic cohort construction
-- Unit of analysis: first ICU stay per hospital admission, LOS >= 24h, age >= 18
\timing on
set work_mem='512MB';

-- ---------- SLE identification ----------
drop table if exists sleci.sle_hadm cascade;
create table sleci.sle_hadm as
select hadm_id,
       max(case when icd_code like 'M3214%' or icd_code like 'M3215%' or icd_code='7104' then 1 else 0 end) lupus_nephritis_icd,
       max(case when icd_code like 'M3219%' or icd_code like 'M3213%' or icd_code like 'M3212%' or icd_code like 'M3211%' then 1 else 0 end) sle_organ_involv,
       max(case when icd_code like 'M320%' then 1 else 0 end) drug_induced_lupus,
       max(case when (icd_version=9 and icd_code like '7100%') or (icd_version=10 and icd_code like 'M32%' and icd_code not like 'M320%') then 1 else 0 end) true_sle
from mimiciv_hosp.diagnoses_icd
where (icd_version=9 and icd_code like '7100%') or (icd_version=10 and icd_code like 'M32%')
group by hadm_id;

-- ---------- base cohort ----------
drop table if exists sleci.cohort cascade;
create table sleci.cohort as
with ranked as (
  select i.*, row_number() over (partition by i.hadm_id order by i.icu_intime) rn
  from mimiciv_derived.icustay_detail i
  join sleci.sle_hadm s on s.hadm_id = i.hadm_id
  where s.true_sle = 1
)
select subject_id, hadm_id, stay_id, gender, admission_age, race, dod,
       admittime, dischtime, icu_intime, icu_outtime, los_icu, los_hospital,
       hospital_expire_flag, hospstay_seq, icustay_seq
from ranked
where rn = 1;
create index on sleci.cohort (stay_id);
create index on sleci.cohort (hadm_id);
analyze sleci.cohort;

select 'step0_first_icu_per_hadm' step, count(*) n from sleci.cohort
union all select 'step1_age_ge18', count(*) from sleci.cohort where admission_age >= 18
union all select 'step2_los_ge_24h', count(*) from sleci.cohort where admission_age >= 18 and los_icu >= 1;

-- ---------- exclusion / sensitivity flags ----------
drop table if exists sleci.excl_flags cascade;
create table sleci.excl_flags as
select c.stay_id,
  max(case when d.icd_code like 'S06%' or d.icd_code like '800%' or d.icd_code like '801%'
            or d.icd_code like '803%' or d.icd_code like '804%' or d.icd_code like '850%'
            or d.icd_code like '851%' or d.icd_code like '852%' or d.icd_code like '853%'
            or d.icd_code like '854%' then 1 else 0 end) tbi,
  max(case when d.icd_code like 'I60%' or d.icd_code like 'I61%' or d.icd_code like 'I62%'
            or d.icd_code like '430%' or d.icd_code like '431%' or d.icd_code like '432%' then 1 else 0 end) ich,
  max(case when d.icd_code like 'I63%' or d.icd_code like '433%' or d.icd_code like '434%' then 1 else 0 end) ischemic_stroke,
  max(case when d.icd_code like 'G00%' or d.icd_code like 'G01%' or d.icd_code like 'G02%'
            or d.icd_code like 'G03%' or d.icd_code like 'G04%' or d.icd_code like 'G05%'
            or d.icd_code like '320%' or d.icd_code like '321%' or d.icd_code like '322%'
            or d.icd_code like '323%' then 1 else 0 end) cns_infection,
  max(case when d.icd_code like 'C71%' or d.icd_code like 'C793%' or d.icd_code like '191%'
            or d.icd_code like '1983%' then 1 else 0 end) brain_tumor
from sleci.cohort c
join mimiciv_hosp.diagnoses_icd d on d.hadm_id = c.hadm_id
group by c.stay_id;
create index on sleci.excl_flags (stay_id);

select sum(tbi) tbi, sum(ich) ich, sum(ischemic_stroke) stroke, sum(cns_infection) cns_inf,
       sum(brain_tumor) brain_tumor,
       sum(case when tbi=1 or ich=1 or ischemic_stroke=1 or cns_infection=1 or brain_tumor=1 then 1 else 0 end) any_primary_brain
from sleci.excl_flags;

-- ---------- final analytic cohort ----------
drop table if exists sleci.cohort_final cascade;
create table sleci.cohort_final as
select c.*, e.tbi, e.ich, e.ischemic_stroke, e.cns_infection, e.brain_tumor,
       case when e.tbi=1 or e.ich=1 or e.ischemic_stroke=1 or e.cns_infection=1 or e.brain_tumor=1
            then 1 else 0 end primary_brain_injury,
       s.lupus_nephritis_icd, s.sle_organ_involv
from sleci.cohort c
left join sleci.excl_flags e using (stay_id)
join sleci.sle_hadm s on s.hadm_id = c.hadm_id
where c.admission_age >= 18 and c.los_icu >= 1;
create index on sleci.cohort_final (stay_id);
create index on sleci.cohort_final (hadm_id);
analyze sleci.cohort_final;

select count(*) n_final, count(distinct subject_id) n_patients,
       sum(primary_brain_injury) n_primary_brain,
       sum(lupus_nephritis_icd) n_lupus_nephritis
from sleci.cohort_final;
