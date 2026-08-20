-- SLECI: MIMIC-IV composite acute brain dysfunction outcome + covariate availability
\timing on

drop table if exists sleci.outcome;
create table sleci.outcome as
with cam as (
  select stay_id,
         max(case when itemid=228332 and value='Positive' then 1 else 0 end) cam_pos,
         max(case when itemid=228332 then 1 else 0 end) cam_assessed
  from sleci.cam_raw group by stay_id),
gcs as (
  select stay_id, min(gcs) gcs_min from mimiciv_derived.gcs group by stay_id),
icd as (
  select c.stay_id,
    max(case when d.icd_code like 'F05%' or d.icd_code in ('2930','29281','29811') then 1 else 0 end) icd_delirium,
    max(case when d.icd_code like 'G93%' or d.icd_code like '3483%' or d.icd_code like 'G31%' then 1 else 0 end) icd_enceph,
    max(case when d.icd_code like 'G40%' or d.icd_code like 'R56%' or d.icd_code like '3459%' or d.icd_code like '78039%' then 1 else 0 end) icd_seizure,
    max(case when d.icd_code like 'M321%' or d.icd_code like '7104%' then 1 else 0 end) icd_sle_organ
  from sleci.cohort_raw c join mimiciv_hosp.diagnoses_icd d on d.hadm_id=c.hadm_id group by 1)
select c.*, coalesce(cam.cam_assessed,0) cam_assessed, coalesce(cam.cam_pos,0) cam_pos,
       g.gcs_min, coalesce(icd.icd_delirium,0) icd_delirium, coalesce(icd.icd_enceph,0) icd_enceph,
       coalesce(icd.icd_seizure,0) icd_seizure
from sleci.cohort_raw c
left join cam using (stay_id) left join gcs g using (stay_id) left join icd using (stay_id);

select count(*) n_stays,
  sum(cam_assessed) cam_assessed, sum(cam_pos) cam_pos,
  sum(icd_delirium) icd_delirium, sum(icd_enceph) icd_enceph, sum(icd_seizure) icd_seizure,
  sum(case when gcs_min<=8 then 1 else 0 end) gcs_le8,
  sum(case when cam_pos=1 or icd_delirium=1 or icd_enceph=1 or gcs_min<=8 then 1 else 0 end) composite_abd
from sleci.outcome;

-- covariate availability on the SLE cohort
select
 (select count(*) from sleci.cohort_raw c join mimiciv_derived.first_day_sofa s using (stay_id)) has_sofa,
 (select count(*) from sleci.cohort_raw c join mimiciv_derived.apsiii  a using (stay_id)) has_apsiii,
 (select count(*) from sleci.cohort_raw c join mimiciv_derived.charlson ch on ch.hadm_id=c.hadm_id) has_charlson,
 (select count(*) from sleci.cohort_raw c join mimiciv_derived.sepsis3 s3 using (stay_id)) has_sepsis3,
 (select count(*) from sleci.cohort_raw c join mimiciv_derived.first_day_lab l using (stay_id)) has_fd_lab,
 (select count(*) from sleci.cohort_raw c join mimiciv_derived.icustay_detail i using (stay_id)) has_detail;

-- demographics
select round(avg(i.admission_age)::numeric,1) mean_age,
       sum(case when i.gender='F' then 1 else 0 end)::float/count(*) pct_female,
       sum(case when i.hospital_expire_flag=1 then 1 else 0 end) hosp_death,
       count(*) n
from sleci.cohort_raw c join mimiciv_derived.icustay_detail i using (stay_id);
