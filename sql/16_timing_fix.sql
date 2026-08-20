-- ============================================================
-- 16_timing_fix.sql
-- 目的：修正时序倒置问题（避免用"结局之后"的信息预测结局）
--   1) mech_vent_24h  : 入 ICU 0-24h 内是否已有创通气/气切
--   2) sepsis3_24h    : 可疑感染时间落在入 ICU 前 24h 至入 ICU 后 24h
--   3) 更新 sleci.analytic，追加上述两列
-- ============================================================

set search_path to sleci, mimiciv_hosp, mimiciv_icu, mimiciv_derived, public;

drop table if exists sleci.timing24 cascade;
create table sleci.timing24 as
with base as (
  select c.stay_id, c.icu_intime as intime
  from sleci.cohort c
),
v24 as (
  select b.stay_id,
         max(case when v.ventilation_status in ('InvasiveVent','Tracheostomy') then 1 else 0 end) as mech_vent_24h,
         max(case when v.ventilation_status in ('NonInvasiveVent','HFNC') then 1 else 0 end) as niv_hfnc_24h
  from base b
  join mimiciv_derived.ventilation v
    on v.stay_id = b.stay_id
   and v.starttime <  b.intime + interval '24 hour'
   and v.endtime   >  b.intime
  group by b.stay_id
),
s24 as (
  select b.stay_id,
         max(case when s.sepsis3 is true
                   and s.suspected_infection_time >= b.intime - interval '24 hour'
                   and s.suspected_infection_time <= b.intime + interval '24 hour'
                  then 1 else 0 end) as sepsis3_24h
  from base b
  join mimiciv_derived.sepsis3 s on s.stay_id = b.stay_id
  group by b.stay_id
)
select b.stay_id,
       coalesce(v24.mech_vent_24h,0) as mech_vent_24h,
       coalesce(v24.niv_hfnc_24h,0)  as niv_hfnc_24h,
       coalesce(s24.sepsis3_24h,0)   as sepsis3_24h
from base b
left join v24 using (stay_id)
left join s24 using (stay_id);

create unique index on sleci.timing24 (stay_id);

-- 与旧口径对比
select count(*) n,
       sum(mech_vent_24h) vent24,
       sum(niv_hfnc_24h)  niv24,
       sum(sepsis3_24h)   sep24
from sleci.timing24;

-- 追加到分析宽表
alter table sleci.analytic drop column if exists mech_vent_24h;
alter table sleci.analytic drop column if exists niv_hfnc_24h;
alter table sleci.analytic drop column if exists sepsis3_24h;
alter table sleci.analytic add column mech_vent_24h int;
alter table sleci.analytic add column niv_hfnc_24h  int;
alter table sleci.analytic add column sepsis3_24h   int;

update sleci.analytic a
set mech_vent_24h = t.mech_vent_24h,
    niv_hfnc_24h  = t.niv_hfnc_24h,
    sepsis3_24h   = t.sepsis3_24h
from sleci.timing24 t
where t.stay_id = a.stay_id;

-- 旧 vs 新 口径交叉核对（仅 CAM 已评估者）
select count(*) n,
       sum(mech_vent)     vent_anytime,
       sum(mech_vent_24h) vent_24h,
       sum(sepsis3)       sepsis_anytime,
       sum(sepsis3_24h)   sepsis_24h
from sleci.analytic
where cam_assessed = 1;
