-- SLECI 15: corrected corticosteroid exposure
--   window: ICU admission -6h to +24h; daily dose = dose_val_rx * doses_per_24_hrs * prednisone factor
\timing on

drop table if exists sleci.steroid cascade;
create table sleci.steroid as
with rx as (
  select c.stay_id,
         p.dose_val_rx::numeric
           * coalesce(p.doses_per_24_hrs, 1)
           * case
               when lower(p.drug) like '%dexamethasone%'      then 6.67
               when lower(p.drug) like '%methylprednisolone%' then 1.25
               when lower(p.drug) like '%hydrocortisone%'     then 0.25
               when lower(p.drug) like '%prednisolone%'       then 1.0
               when lower(p.drug) like '%prednisone%'         then 1.0
               when lower(p.drug) like '%cortisone%'          then 0.20
               when lower(p.drug) like '%betamethasone%'      then 8.33
             end pred_eq,
         p.starttime, p.stoptime
  from sleci.cohort_final c
  join mimiciv_hosp.prescriptions p on p.hadm_id = c.hadm_id
  where (lower(p.drug) like '%prednis%' or lower(p.drug) like '%methylprednisolone%'
         or lower(p.drug) like '%hydrocortisone%' or lower(p.drug) like '%dexamethasone%'
         or lower(p.drug) like '%cortisone%' or lower(p.drug) like '%betamethasone%')
    and lower(p.drug) not like '%cream%' and lower(p.drug) not like '%ophth%'
    and lower(p.drug) not like '%otic%'  and lower(p.drug) not like '%topical%'
    and lower(p.drug) not like '%inhal%' and lower(p.drug) not like '%nasal%'
    and (p.route is null or upper(p.route) in
         ('PO','PO/NG','NG','ORAL','IV','IV DRIP','IV BOLUS','IVPCA','PB','IM','PO/PR','PR'))
    and p.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$'
    and p.starttime >= c.icu_intime - interval '6 hour'
    and p.starttime <= c.icu_intime + interval '24 hour'
),
rx_all as (  -- whole hospital stay, for "any steroid" flag
  select c.stay_id, count(*) n_rx
  from sleci.cohort_final c
  join mimiciv_hosp.prescriptions p on p.hadm_id = c.hadm_id
  where (lower(p.drug) like '%prednis%' or lower(p.drug) like '%methylprednisolone%'
         or lower(p.drug) like '%hydrocortisone%' or lower(p.drug) like '%dexamethasone%')
    and lower(p.drug) not like '%cream%' and lower(p.drug) not like '%ophth%'
    and lower(p.drug) not like '%otic%'  and lower(p.drug) not like '%topical%'
    and lower(p.drug) not like '%inhal%' and lower(p.drug) not like '%nasal%'
  group by c.stay_id
)
select c.stay_id,
  case when ra.n_rx > 0 then 1 else 0 end                      steroid_hosp_any,
  case when coalesce(sum(rx.pred_eq),0) > 0 then 1 else 0 end  steroid_24h,
  coalesce(sum(rx.pred_eq), 0)                                 pred_eq_24h,
  case when coalesce(sum(rx.pred_eq),0) >= 250 then 1 else 0 end pulse_steroid,
  case when coalesce(sum(rx.pred_eq),0) = 0 then 'none'
       when sum(rx.pred_eq) < 30  then 'low(<30)'
       when sum(rx.pred_eq) < 100 then 'moderate(30-100)'
       when sum(rx.pred_eq) < 250 then 'high(100-250)'
       else 'pulse(>=250)' end                                 steroid_cat
from sleci.cohort_final c
left join rx     on rx.stay_id = c.stay_id
left join rx_all ra on ra.stay_id = c.stay_id
group by c.stay_id, ra.n_rx;
create index on sleci.steroid (stay_id);
analyze sleci.steroid;

select steroid_cat, count(*) n,
       round(100.0*count(*)/sum(count(*)) over (),1) pct,
       round(avg(pred_eq_24h),1) mean_dose
from sleci.steroid group by 1 order by 2 desc;

select sum(steroid_hosp_any) hosp_any, sum(steroid_24h) icu24h_any,
       round(avg(pred_eq_24h),1) mean_all,
       round(percentile_cont(0.5) within group (order by pred_eq_24h)::numeric,1) median_all,
       round(avg(case when pred_eq_24h>0 then pred_eq_24h end),1) mean_exposed
from sleci.steroid;
