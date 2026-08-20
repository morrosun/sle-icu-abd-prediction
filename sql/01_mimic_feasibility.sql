-- SLECI: MIMIC-IV feasibility — SLE ICU cohort & acute brain dysfunction outcome availability
\timing on
set work_mem = '512MB';

-- Step 1: pull ALL CAM-ICU / RASS / delirium rows (itemid index is efficient), then join
drop table if exists sleci.cam_all;
create table sleci.cam_all as
select stay_id, charttime, itemid, value
from mimiciv_icu.chartevents
where itemid in (228332,228300,228301,228302,228303,228334,228335,228336,228337,
                 229324,229325,229326,228688,228096);
create index on sleci.cam_all (stay_id);
analyze sleci.cam_all;

drop table if exists sleci.cam_raw;
create table sleci.cam_raw as
select a.* from sleci.cam_all a join sleci.cohort_raw c using (stay_id);
create index on sleci.cam_raw (stay_id);
analyze sleci.cam_raw;

select count(*) n_rows, count(distinct stay_id) n_stays from sleci.cam_raw;

with cam as (
  select stay_id,
         max(case when itemid=228332 and value='Positive' then 1 else 0 end) cam_pos,
         max(case when itemid=228332 then 1 else 0 end) cam_assessed,
         max(case when itemid=228688 and value in ('Yes','Positive') then 1 else 0 end) delirium_flag
  from sleci.cam_raw group by stay_id)
select count(*) sle_icustays,
       sum(coalesce(cam_assessed,0)) with_camicu,
       sum(coalesce(cam_pos,0))      camicu_positive,
       sum(coalesce(delirium_flag,0)) delirium_item_pos
from sleci.cohort_raw left join cam using (stay_id);
