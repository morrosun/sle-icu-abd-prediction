-- SLECI 14: assemble analytic wide table
\timing on

drop table if exists sleci.analytic cascade;
create table sleci.analytic as
select
  c.stay_id, c.subject_id, c.hadm_id,
  c.admission_age age,
  case when c.gender='F' then 1 else 0 end female,
  case when c.race ilike '%white%' then 'White'
       when c.race ilike '%black%' then 'Black'
       when c.race ilike '%hispanic%' or c.race ilike '%latino%' then 'Hispanic'
       when c.race ilike '%asian%' then 'Asian'
       else 'Other/Unknown' end race_grp,
  c.los_icu, c.los_hospital,
  c.lupus_nephritis_icd, c.sle_organ_involv, c.primary_brain_injury,
  c.tbi, c.ich, c.ischemic_stroke, c.cns_infection, c.brain_tumor,
  -- comorbidities
  cm.htn, cm.diabetes, cm.ckd, cm.esrd_dialysis, cm.epilepsy_hx, cm.dementia_hx,
  cm.depression_hx, cm.alcohol_abuse, cm.aps_antiphospholipid, cm.chf, cm.copd,
  cm.malignancy, cm.anemia_dx, cm.itp,
  -- severity & physiology
  cv.sofa_24h, cv.sofa_cns, cv.sofa_resp, cv.sofa_coag, cv.sofa_liver, cv.sofa_cv, cv.sofa_renal,
  cv.apsiii, cv.sapsii, cv.oasis, cv.charlson,
  cv.heart_rate_mean, cv.sbp_mean, cv.mbp_min, cv.resp_rate_mean,
  cv.temperature_max, cv.temperature_min, cv.spo2_min,
  cv.wbc_max, cv.hemoglobin_min, cv.platelets_min, cv.albumin_min,
  cv.creatinine_max, cv.bun_max, cv.sodium_min, cv.sodium_max, cv.potassium_max,
  cv.bicarbonate_min, cv.aniongap_max, cv.glucose_max, cv.inr_max, cv.ptt_max,
  cv.bilirubin_total_max, cv.alt_max, cv.ast_max, cv.calcium_min,
  cv.abs_lymphocytes_min, cv.abs_neutrophils_max,
  cv.lactate_max, cv.ph_min, cv.pao2fio2ratio_min, cv.crp_max_24h,
  cv.weight_admit, cv.bmi, cv.uo_24h,
  -- treatments
  st.steroid_hosp_any, st.steroid_24h, st.pred_eq_24h, st.pulse_steroid, st.steroid_cat,
  im.hcq, im.mmf, im.aza, im.cyc, im.cni, im.biologic,
  sd.benzo_24h, sd.propofol_24h, sd.dexmed_24h, sd.opioid_24h,
  ap.haloperidol, ap.atypical_ap,
  sup.sepsis3, sup.rrt_24h, sup.vaso_24h,
  -- outcomes
  o.cam_assessed, o.cam_pos, o.cam_first_pos_hr, o.cam_pos_24h, o.cam_pos_after24h,
  o.n_cam_assess, o.n_cam_pos, o.n_cam_uta, o.rass_min, o.rass_max,
  o.gcs_min_icu, o.gcs_min_24h,
  o.icd_delirium, o.icd_encephalopathy, o.icd_seizure, o.icd_psychosis,
  o.composite_abd, o.mech_vent, o.vent_hours,
  o.hospital_expire_flag, o.icu_expire_flag, o.mort_28d, o.mort_90d
from sleci.cohort_final c
left join sleci.comorbid   cm using (stay_id)
left join sleci.covariates cv using (stay_id)
left join sleci.steroid    st using (stay_id)
left join sleci.immunosupp im using (stay_id)
left join sleci.sedation   sd using (stay_id)
left join sleci.antipsych  ap using (stay_id)
left join sleci.support    sup using (stay_id)
left join sleci.outcomes   o  using (stay_id);
create index on sleci.analytic (stay_id);
analyze sleci.analytic;

select count(*) n, count(distinct subject_id) pts,
       sum(cam_assessed) assessed, sum(cam_pos) cam_pos, sum(composite_abd) composite
from sleci.analytic;
