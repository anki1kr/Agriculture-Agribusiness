# Week 4 - Predictive Modeling & Yield Forecasting

**Track:** Junior Machine Learning Data Analyst - Agriculture & Agribusiness  
**Author:** Ankit Kumar  

---

### Deliverable Overview
- **Document Title:** `Predictive_Modeling_for_Agriculture_Applications.docx`
- **Model Training Script:** `train_agronomic_yield_model.py`
- **Focus:** Leave-One-Year-Out (LOYO) cross-validation, LightGBM with monotonic agronomic constraints.

### Document Structure & Core Sections
1. **Problem Statement & Agronomic Background:** Pre-harvest yield forecasting (t/ha), technology trend de-trending, nonlinear heat stress during reproductive flowering.
2. **Multi-Source Feature Engineering & Agronomic Transformations:**
   - Growing Degree Days (GDD) & Extreme Degree Days (EDD > 30°C).
   - Atmospheric Vapor Pressure Deficit (VPD) and 90-day SPEI drought indices.
   - MODIS MOD13Q1 canopy phenology (peak NDVI and post-peak senescence slope).
   - 28-feature catalog covering thermal, moisture, soil (ISRIC 250m), and technological drivers.
3. **Model Selection Architecture & Theoretical Rationale:**
   - Trade-off matrix: Ridge Baseline, Random Forest, LightGBM (Champion), SAR-SEM Spatial Econometrics.
   - Domain monotonic constraints injected directly into LightGBM loss objective to prevent spurious positive splines on heat stress.
4. **Model Training & Spatial-Temporal Validation Strategy:**
   - Preventing spatial data leakage: Leave-One-Year-Out (LOYO) and 150 km Agro-Climatic Zone buffer cross-validation (Roberts et al., 2017).
   - Complete production Python training pipeline script using LightGBM.
5. **Comprehensive Evaluation Metrics & Validation Benchmarks:**
   - RMSE, MAE, MAPE, Directional Anomaly Accuracy (DAA), and Asymmetric Downside Loss.
   - Out-of-sample performance table across 5 validation years (2019–2023) across 320 districts.
6. **Commercial Application Scenarios & Academic Citations:**
   - Parametric area-yield index crop insurance underwriting.
   - Industrial grain procurement hedging and storage scheduling.
   - National food security emergency buffer allocation.
   - Formal bibliography: USDA NASS API, ECMWF ERA5-Land, NASA MODIS, Schlenker & Roberts (2009 PNAS), Lobell et al. (2011 Science), Roberts et al. (2017 AJAE).

### Status
- [x] Deliverable and documentation completed.