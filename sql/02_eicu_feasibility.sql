-- SLECI: eICU-CRD feasibility
\timing on
create schema if not exists sleci;

-- SLE identification: diagnosis string / ICD, plus antimalarial (hydroxychloroquine) evidence
drop table if exists sleci.sle_dx;
create table sleci.sle_dx as
select distinct patientunitstayid from eicu_crd.diagnosis
where lower(diagnosisstring) like '%systemic lupus%' or icd9code like '710.0%' or icd9code like '%M32.1%' or icd9code like '%M32.9%';

drop table if exists sleci.sle_drug;
create table sleci.sle_drug as
select distinct patientunitstayid from eicu_crd.medication
where lower(drugname) like '%hydroxychloroquine%' or lower(drugname) like '%plaquenil%'
   or lower(drugname) like '%mycophenolate%' or lower(drugname) like '%cellcept%'
   or lower(drugname) like '%azathioprine%' or lower(drugname) like '%imuran%'
   or lower(drugname) like '%belimumab%' or lower(drugname) like '%cyclophosphamide%';

drop table if exists sleci.sle_adm;
create table sleci.sle_adm as
select distinct patientunitstayid from eicu_crd.admissiondrug
where lower(drugname) like '%hydroxychloroquine%' or lower(drugname) like '%plaquenil%';

select (select count(*) from sleci.sle_dx) dx_only,
       (select count(*) from sleci.sle_drug) any_immunosupp,
       (select count(*) from sleci.sle_adm) adm_hcq,
       (select count(*) from (select patientunitstayid from sleci.sle_dx
          intersect select patientunitstayid from
            (select patientunitstayid from sleci.sle_drug union select patientunitstayid from sleci.sle_adm) u) x) dx_plus_drug;

-- Outcome availability: delirium-related nursecharting items
drop table if exists sleci.nc_delirium;
create table sleci.nc_delirium as
select patientunitstayid, nursingchartoffset, nursingchartcelltypevallabel lbl,
       nursingchartcelltypevalname nm, nursingchartvalue val
from eicu_crd.nursecharting
where nursingchartcelltypevallabel in ('Delirium Scale/Score','Sedation Scale/Score/Level','Glasgow coma score')
   or nursingchartcelltypevalname in ('Delirium Score','Delirium Scale','Value','GCS Total','Sedation Score');
create index on sleci.nc_delirium (patientunitstayid);
analyze sleci.nc_delirium;

select lbl, nm, count(*) n, count(distinct patientunitstayid) n_stay
from sleci.nc_delirium group by 1,2 order by n desc limit 25;
