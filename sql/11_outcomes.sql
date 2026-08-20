-- SLECI 11: outcomes
\timing on
set work_mem='512MB';

drop table if exists sleci.outcomes cascade;
create table sleci.outcomes as
with cam as (
  select ce.stay_id, ce.charttime, ce.itemid, ce.value, c.icu_intime,
         extract(epoch from (ce.charttime - c.icu_intime))/3600.0 hr_from_intime
  from sleci.cam_all ce join sleci.cohort_final c using (stay_id)
),
cam_sum as (
  select stay_id,
    max(case when itemid=228332 then 1 else 0 end)                                  cam_assessed,
    max(case when itemid=228332 and value='Positive' then 1 else 0 end)             cam_pos,
    min(case when itemid=228332 and value='Positive' then hr_from_intime end)       cam_first_pos_hr,
    max(case when itemid=228332 and value='Positive' and hr_from_intime<=24 then 1 else 0 end) cam_pos_24h,
    max(case when itemid=228332 and value='Positive' and hr_from_intime> 24 then 1 else 0 end) cam_pos_after24h,
    sum(case when itemid=228332 and value='Positive' then 1 else 0 end)             n_cam_pos,
    sum(case when itemid=228332 then 1 else 0 end)                                  n_cam_assess,
    sum(case when itemid=228332 and value='UTA' then 1 else 0 end)                  n_cam_uta,
    min(case when itemid=228096 and value ~ '^-?[0-9]' then substring(value from '^-?[0-9]+')::int end) rass_min,
    max(case when itemid=228096 and value ~ '^-?[0-9]' then substring(value from '^-?[0-9]+')::int end) rass_max
  from cam group by stay_id
),
gcs as (
  select g.stay_id, min(g.gcs) gcs_min_icu,
         min(case when extract(epoch from (g.charttime - c.icu_intime))/3600.0 <= 24 then g.gcs end) gcs_min_24h
  from mimiciv_derived.gcs g join sleci.cohort_final c using (stay_id) group by g.stay_id
),
icd as (
  select c.stay_id,
    max(case when d.icd_code like 'F05%' or d.icd_code in ('2930','29281','29811','29310','29311') then 1 else 0 end) icd_delirium,
    max(case when d.icd_code like 'G93%' or d.icd_code like '3483%' or d.icd_code like 'G311%' or d.icd_code like 'G3289%' then 1 else 0 end) icd_encephalopathy,
    max(case when d.icd_code like 'G40%' or d.icd_code like 'R56%' or d.icd_code like '3459%' or d.icd_code like '78039%' or d.icd_code like '3453%' then 1 else 0 end) icd_seizure,
    max(case when d.icd_code like 'F2%' or d.icd_code like 'F31%' or d.icd_code like '295%' or d.icd_code like '298%' then 1 else 0 end) icd_psychosis
  from sleci.cohort_final c join mimiciv_hosp.diagnoses_icd d on d.hadm_id=c.hadm_id group by c.stay_id
),
vent as (
  select v.stay_id,
    max(case when v.ventilation_status in ('InvasiveVent','Tracheostomy') then 1 else 0 end) mech_vent,
    sum(case when v.ventilation_status in ('InvasiveVent','Tracheostomy')
             then extract(epoch from (v.endtime - v.starttime))/3600.0 else 0 end) vent_hours
  from mimiciv_derived.ventilation v join sleci.cohort_final c using (stay_id) group by v.stay_id
)
select c.stay_id, c.subject_id, c.hadm_id,
  coalesce(cs.cam_assessed,0) cam_assessed,
  coalesce(cs.cam_pos,0)      cam_pos,
  cs.cam_first_pos_hr,
  coalesce(cs.cam_pos_24h,0)  cam_pos_24h,
  coalesce(cs.cam_pos_after24h,0) cam_pos_after24h,
  coalesce(cs.n_cam_pos,0) n_cam_pos, coalesce(cs.n_cam_assess,0) n_cam_assess,
  coalesce(cs.n_cam_uta,0) n_cam_uta,
  cs.rass_min, cs.rass_max,
  g.gcs_min_icu, g.gcs_min_24h,
  coalesce(i.icd_delirium,0) icd_delirium, coalesce(i.icd_encephalopathy,0) icd_encephalopathy,
  coalesce(i.icd_seizure,0) icd_seizure, coalesce(i.icd_psychosis,0) icd_psychosis,
  case when coalesce(cs.cam_pos,0)=1 or coalesce(i.icd_delirium,0)=1
            or coalesce(i.icd_encephalopathy,0)=1 or coalesce(g.gcs_min_icu,15)<=8
       then 1 else 0 end composite_abd,
  coalesce(v.mech_vent,0) mech_vent, coalesce(v.vent_hours,0) vent_hours,
  c.los_icu, c.los_hospital, c.hospital_expire_flag,
  case when c.dod is not null and c.dod <= (c.icu_intime + interval '28 day')::date then 1 else 0 end mort_28d,
  case when c.dod is not null and c.dod <= (c.icu_intime + interval '90 day')::date then 1 else 0 end mort_90d,
  case when c.dod is not null and c.dod <= c.icu_outtime::date then 1 else 0 end icu_expire_flag
from sleci.cohort_final c
left join cam_sum cs using (stay_id)
left join gcs g using (stay_id)
left join icd i using (stay_id)
left join vent v using (stay_id);
create index on sleci.outcomes (stay_id);
analyze sleci.outcomes;

select count(*) n,
  sum(cam_assessed) cam_assessed, sum(cam_pos) cam_pos,
  sum(cam_pos_24h) pos_within24h, sum(cam_pos_after24h) pos_after24h,
  sum(composite_abd) composite,
  sum(icd_delirium) icd_del, sum(icd_encephalopathy) icd_enc, sum(icd_seizure) icd_sz, sum(icd_psychosis) icd_psy,
  sum(case when gcs_min_icu<=8 then 1 else 0 end) gcs_le8,
  sum(mech_vent) vent, sum(hospital_expire_flag) hosp_death, sum(mort_28d) d28, sum(mort_90d) d90
from sleci.outcomes;

-- incident delirium cohort (exclude prevalent delirium in first 24h)
select count(*) n_incident_cohort, sum(cam_pos_after24h) n_incident_delirium
from sleci.outcomes where cam_pos_24h = 0;
