-- SLECI 13: treatment exposures within first 24h of ICU (and whole ICU stay where noted)
\timing on
set work_mem='512MB';

-- ---------- corticosteroids: prednisone-equivalent daily dose ----------
drop table if exists sleci.steroid_rx cascade;
create table sleci.steroid_rx as
select c.stay_id, p.drug, p.dose_val_rx, p.dose_unit_rx, p.route, p.starttime, p.stoptime,
  case
    when lower(p.drug) like '%dexamethasone%'      then 6.67
    when lower(p.drug) like '%methylprednisolone%' then 1.25
    when lower(p.drug) like '%hydrocortisone%'     then 0.25
    when lower(p.drug) like '%prednisolone%'       then 1.0
    when lower(p.drug) like '%prednisone%'         then 1.0
    when lower(p.drug) like '%cortisone%'          then 0.20
    when lower(p.drug) like '%betamethasone%'      then 8.33
  end pred_factor
from sleci.cohort_final c
join mimiciv_hosp.prescriptions p on p.hadm_id = c.hadm_id
where (lower(p.drug) like '%prednis%' or lower(p.drug) like '%methylprednisolone%'
       or lower(p.drug) like '%hydrocortisone%' or lower(p.drug) like '%dexamethasone%'
       or lower(p.drug) like '%cortisone%' or lower(p.drug) like '%betamethasone%')
  and (p.route is null or upper(p.route) in ('PO','PO/NG','IV','IV DRIP','IVPCA','PO/PR','ORAL','NG','IM','PB','IV BOLUS'))
  and p.starttime <= c.icu_outtime and coalesce(p.stoptime, p.starttime) >= c.icu_intime - interval '1 day';
create index on sleci.steroid_rx (stay_id);

drop table if exists sleci.steroid cascade;
create table sleci.steroid as
select c.stay_id,
  max(case when s.stay_id is not null then 1 else 0 end) steroid_any,
  coalesce(sum(case when s.pred_factor is not null
                     and s.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$'
                     and s.starttime <= c.icu_intime + interval '24 hour'
                    then s.dose_val_rx::numeric * s.pred_factor else 0 end), 0) pred_eq_24h,
  max(case when s.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$' and s.pred_factor is not null
                 and s.dose_val_rx::numeric * s.pred_factor >= 250 then 1 else 0 end) pulse_steroid
from sleci.cohort_final c left join sleci.steroid_rx s using (stay_id)
group by c.stay_id;
create index on sleci.steroid (stay_id);

-- ---------- immunosuppressants / antimalarials ----------
drop table if exists sleci.immunosupp cascade;
create table sleci.immunosupp as
select c.stay_id,
  max(case when lower(p.drug) like '%hydroxychloroquine%' or lower(p.drug) like '%plaquenil%' then 1 else 0 end) hcq,
  max(case when lower(p.drug) like '%mycophenolate%' or lower(p.drug) like '%cellcept%' or lower(p.drug) like '%myfortic%' then 1 else 0 end) mmf,
  max(case when lower(p.drug) like '%azathioprine%' or lower(p.drug) like '%imuran%' then 1 else 0 end) aza,
  max(case when lower(p.drug) like '%cyclophosphamide%' then 1 else 0 end) cyc,
  max(case when lower(p.drug) like '%tacrolimus%' or lower(p.drug) like '%cyclosporin%' then 1 else 0 end) cni,
  max(case when lower(p.drug) like '%rituximab%' or lower(p.drug) like '%belimumab%' then 1 else 0 end) biologic,
  max(case when lower(p.drug) like '%methotrexate%' then 1 else 0 end) mtx
from sleci.cohort_final c
left join mimiciv_hosp.prescriptions p on p.hadm_id = c.hadm_id
group by c.stay_id;
create index on sleci.immunosupp (stay_id);

-- ---------- sedatives / analgesics / antipsychotics (first 24h, ICU inputevents) ----------
drop table if exists sleci.sed_items cascade;
create table sleci.sed_items as
select itemid, label,
  case
    when lower(label) like '%midazolam%'      then 'benzo'
    when lower(label) like '%lorazepam%'      then 'benzo'
    when lower(label) like '%diazepam%'       then 'benzo'
    when lower(label) like '%propofol%'       then 'propofol'
    when lower(label) like '%dexmedetomidine%' then 'dexmed'
    when lower(label) like '%precedex%'       then 'dexmed'
    when lower(label) like '%fentanyl%'       then 'opioid'
    when lower(label) like '%morphine%'       then 'opioid'
    when lower(label) like '%hydromorphone%'  then 'opioid'
    when lower(label) like '%ketamine%'       then 'ketamine'
  end cls
from mimiciv_icu.d_items
where lower(label) similar to '%(midazolam|lorazepam|diazepam|propofol|dexmedetomidine|precedex|fentanyl|morphine|hydromorphone|ketamine)%';

drop table if exists sleci.sedation cascade;
create table sleci.sedation as
with ie as (
  select c.stay_id, si.cls, ie.amount, ie.starttime
  from sleci.cohort_final c
  join mimiciv_icu.inputevents ie on ie.stay_id = c.stay_id
  join sleci.sed_items si on si.itemid = ie.itemid
  where si.cls is not null and ie.starttime <= c.icu_intime + interval '24 hour'
)
select c.stay_id,
  max(case when ie.cls='benzo' then 1 else 0 end)    benzo_24h,
  max(case when ie.cls='propofol' then 1 else 0 end) propofol_24h,
  max(case when ie.cls='dexmed' then 1 else 0 end)   dexmed_24h,
  max(case when ie.cls='opioid' then 1 else 0 end)   opioid_24h,
  max(case when ie.cls='ketamine' then 1 else 0 end) ketamine_24h
from sleci.cohort_final c left join ie using (stay_id)
group by c.stay_id;
create index on sleci.sedation (stay_id);

drop table if exists sleci.antipsych cascade;
create table sleci.antipsych as
select c.stay_id,
  max(case when lower(p.drug) like '%haloperidol%' or lower(p.drug) like '%haldol%' then 1 else 0 end) haloperidol,
  max(case when lower(p.drug) like '%quetiapine%' or lower(p.drug) like '%olanzapine%'
            or lower(p.drug) like '%risperidone%' or lower(p.drug) like '%ziprasidone%' then 1 else 0 end) atypical_ap
from sleci.cohort_final c
left join mimiciv_hosp.prescriptions p on p.hadm_id = c.hadm_id
group by c.stay_id;
create index on sleci.antipsych (stay_id);

-- ---------- organ support & infection ----------
drop table if exists sleci.support cascade;
create table sleci.support as
select c.stay_id,
  case when s3.stay_id is not null then 1 else 0 end sepsis3,
  coalesce(rrt.rrt_24h,0) rrt_24h,
  coalesce(va.vaso_24h,0) vaso_24h
from sleci.cohort_final c
left join mimiciv_derived.sepsis3 s3 using (stay_id)
left join (select c2.stay_id, 1 rrt_24h from sleci.cohort_final c2
             join mimiciv_derived.rrt r on r.stay_id=c2.stay_id
            where r.charttime <= c2.icu_intime + interval '24 hour' and r.dialysis_active=1
            group by c2.stay_id) rrt using (stay_id)
left join (select c3.stay_id, 1 vaso_24h from sleci.cohort_final c3
             join mimiciv_derived.vasoactive_agent v on v.stay_id=c3.stay_id
            where v.starttime <= c3.icu_intime + interval '24 hour'
              and (v.norepinephrine is not null or v.epinephrine is not null or v.dopamine is not null
                   or v.phenylephrine is not null or v.vasopressin is not null)
            group by c3.stay_id) va using (stay_id);
create index on sleci.support (stay_id);

select sum(steroid_any) steroid, round(avg(pred_eq_24h),1) mean_pred_eq, sum(pulse_steroid) pulse from sleci.steroid;
select sum(hcq) hcq, sum(mmf) mmf, sum(aza) aza, sum(cyc) cyc, sum(cni) cni, sum(biologic) bio from sleci.immunosupp;
select sum(benzo_24h) benzo, sum(propofol_24h) propofol, sum(dexmed_24h) dexmed, sum(opioid_24h) opioid from sleci.sedation;
select sum(haloperidol) halo, sum(atypical_ap) atyp from sleci.antipsych;
select sum(sepsis3) sepsis, sum(rrt_24h) rrt, sum(vaso_24h) vaso from sleci.support;
