-- SLECI 26: first benzodiazepine exposure hour relative to ICU admission (MIMIC-IV)
-- for the DEV cohort (cam_assessed & no primary brain injury), to enable a
-- reverse-causality landmark / early-window sensitivity analysis.
drop table if exists dev_stay;
create temp table dev_stay(stay_id bigint);
\copy dev_stay(stay_id) from 'data/dev_stays.csv' with (format csv, header true);

-- first benzo hour within first 24h of ICU stay
select ds.stay_id,
       min(extract(epoch from ie.starttime - ic.intime)/3600.0) as benzo_first_hour
from dev_stay ds
join mimiciv_icu.icustays ic on ic.stay_id = ds.stay_id
join mimiciv_icu.inputevents ie on ie.stay_id = ds.stay_id
join mimiciv_icu.d_items di on di.itemid = ie.itemid
where lower(di.label) ~ 'midazolam|lorazepam|diazepam'
  and ie.starttime >= ic.intime
  and ie.starttime <= ic.intime + interval '24 hour'
group by ds.stay_id
order by ds.stay_id;
