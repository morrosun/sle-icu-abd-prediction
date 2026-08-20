-- SLECI: eICU SLE cohort x delirium outcome availability
\timing on

drop table if exists sleci.cohort_raw;
create table sleci.cohort_raw as
select distinct patientunitstayid from (
  select patientunitstayid from sleci.sle_dx
  union
  select patientunitstayid from sleci.sle_adm
) u;
create index on sleci.cohort_raw (patientunitstayid);
analyze sleci.cohort_raw;

drop table if exists sleci.delirium_nc;
create table sleci.delirium_nc as
select patientunitstayid, nursingchartoffset,
       nursingchartcelltypevalname nm, nursingchartvalue val
from eicu_crd.nursecharting
where nursingchartcelltypevallabel in ('Delirium Scale/Score','Symptoms of Delirium Present');
create index on sleci.delirium_nc (patientunitstayid);
analyze sleci.delirium_nc;

-- coverage in whole eICU
select count(distinct patientunitstayid) stays_with_delirium_assessment from sleci.delirium_nc;

-- SLE cohort size and outcome
with d as (
  select patientunitstayid,
         max(case when (nm='Delirium Score' and upper(val) in ('YES')) then 1
                  when (nm='Delirium Score' and val ~ '^[0-9]+$' and val::int >= 4) then 1
                  when (nm='Value' and upper(val)='YES') then 1 else 0 end) delirium_pos,
         max(case when nm in ('Delirium Score','Value') then 1 else 0 end) assessed
  from sleci.delirium_nc group by 1)
select count(*) sle_stays,
       sum(coalesce(d.assessed,0)) with_assessment,
       sum(coalesce(d.delirium_pos,0)) delirium_positive
from sleci.cohort_raw c left join d using (patientunitstayid);

-- ICD/diagnosis-based acute brain dysfunction in SLE cohort
select count(distinct c.patientunitstayid) sle_stays,
  count(distinct case when lower(dx.diagnosisstring) like '%delirium%' then c.patientunitstayid end) dx_delirium,
  count(distinct case when lower(dx.diagnosisstring) like '%encephalopath%' then c.patientunitstayid end) dx_enceph,
  count(distinct case when lower(dx.diagnosisstring) like '%change in mental status%' or lower(dx.diagnosisstring) like '%obtundation%' or lower(dx.diagnosisstring) like '%coma%' then c.patientunitstayid end) dx_ams,
  count(distinct case when lower(dx.diagnosisstring) like '%seizure%' then c.patientunitstayid end) dx_seizure
from sleci.cohort_raw c left join eicu_crd.diagnosis dx using (patientunitstayid);
