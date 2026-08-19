# Week 5 - Model Evaluation & Spatial-Temporal Validation

**Track:** Junior Machine Learning Data Analyst - Agriculture & Agribusiness  
**Author:** Ankit Kumar  

---

### Deliverable Overview
- **Document Title:** `Model_Evaluation_and_Interpretation_in_Agricultural_Analytics.docx`
- **Diagnostics Script:** `model_evaluation_and_diagnostics.py`
- **Focus:** Moran's I spatial residual autocorrelation, TreeSHAP feature attributions, asymmetric downside loss.

### Document Structure & Core Sections
1. **Executive Summary & Evaluation Philosophy:** Evaluating beyond naive metrics; spatial error spillovers across contiguous agricultural corridors; tail-risk catastrophic climate years.
2. **Agricultural Evaluation Metrics & Benchmark Formulations:**
   - Physical RMSE (t/ha), MAE, relative scale-invariant MAPE.
   - Directional Anomaly Accuracy (DAA) relative to 10-year Olympic baselines.
   - Spatial Residual Moran's I ($I < 0.05, p > 0.05$) to audit omitted spatial variables.
   - Asymmetric Downside Risk Loss penalizing optimistic forecasts during drought at 2.5x.
   - Out-of-year benchmark comparison across models (2019–2023).
3. **Spatial & Temporal Residual Diagnostic Process:**
   - 4-step diagnostic audit: Temporal LOYO, Spatial Queen contiguity weight matrices, Breusch-Pagan heteroscedasticity, phenological lead-time convergence (60 to 15 days pre-harvest).
   - Production Python diagnostic script integrating PySAL Moran's I and TreeSHAP.
4. **Agronomic Interpretation Framework & Feature Attribution:**
   - Validating non-linear biological response curves with TreeSHAP and Partial Dependence.
   - Confirming the empirical 30°C extreme heat threshold (Schlenker & Roberts, 2009).
   - Atmospheric VPD cutoff at 2.2 kPa and available water capacity (AWC) buffer interactions.
5. **Critical Limitations & Agricultural Data Artifacts:**
   - Cloud occlusion in optical MODIS composites.
   - Omission of micro-level farm management covariates (cultivars, sowing dates).
   - Reanalysis boundary discretization and topographic microclimate smoothing.
   - Extrapolation bounds under non-stationary extreme climate shifts.
6. **Concrete Future Improvement Roadmap:**
   - Sentinel-1 dual-pol Synthetic Aperture Radar (SAR) for cloud-penetrating moisture tracking.
   - Hybrid Physics-Informed ML coupling biophysical crop models (DSSAT / APSIM) with GBDTs.
   - Spatial Graph Neural Networks (GNNs) over watershed contiguity graphs.
   - Conformal prediction intervals for distribution-free risk quantification.

### Status
- [x] Deliverable and documentation completed.