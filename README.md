# Agriculture & Agribusiness ML Analytics
**Author:** Ankit Kumar | Junior Machine Learning Data Analyst  
**Track:** 6-Week Agriculture Analytics Internship

Pre-harvest crop yield forecasting engine across 320 agricultural districts (2009–2023). Fuses NASA MODIS satellite NDVI, ECMWF ERA5-Land weather reanalysis, and ISRIC SoilGrids data. Uses biologically constrained LightGBM with Leave-One-Year-Out (LOYO) spatial cross-validation to eliminate weather leakage, reducing out-of-sample MAPE from 15.4% to 7.4%.

---

### Quick Start
```bash
pip install -r requirements.txt

# Run predictive yield model
python "Week 4 - Predictive Modeling & Yield Forecasting/train_agronomic_yield_model.py"

# Run executive procurement scorecard
python "Week 6 - Pipeline Deployment & Capstone Delivery/capstone_pipeline_and_dashboard.py"
```

---

### Repository Structure & Deliverables

| Week | Phase | Operational Pipeline | Report Deliverable |
| :--- | :--- | :--- | :--- |
| **Week 1** | Problem Formulation & Baseline | `problem_formulation.py` | `Strategic_Planning_and_Data_Problem_Definition.docx` |
| **Week 2** | Ingestion & Cloud Cleaning | `satellite_ndvi_cleaner.py` | `Agricultural_Data_Acquisition_and_Cleaning_Methodology.docx` |
| **Week 3** | Agronomic EDA (GDD, EDD, VPD) | `agronomic_eda_and_feature_engineering.py` | `Exploratory_Data_Analysis_and_Visualization_in_Agriculture.docx` |
| **Week 4** | Predictive Yield Modeling | `train_agronomic_yield_model.py` | `Predictive_Modeling_for_Agriculture_Applications.docx` |
| **Week 5** | Spatial Diagnostics (Moran's I) | `model_evaluation_and_diagnostics.py` | `Model_Evaluation_and_Interpretation_in_Agricultural_Analytics.docx` |
| **Week 6** | Executive Capstone Scorecard | `capstone_pipeline_and_dashboard.py` | `Comprehensive_Project_Report_and_Presentation_Strategy.docx` |

All benchmark datasets and exported predictions reside in `data/`. Detailed reports and analysis documents are organized in each week's folder.

---

### Empirical Highlights
- **Error Reduction:** Out-of-sample MAPE dropped from 15.4% (5-year baseline) to 7.4% (LightGBM).
- **Directional Anomaly:** 86.5% accuracy in detecting boom vs. drought seasons 30–45 days pre-harvest.
- **Spatial Independence:** Residual Moran's $I = 0.0037$ ($p = 0.28$), confirming zero spatial autocorrelation leakage.
- **Thermal Inflection:** $-0.0084$ t/ha yield loss per Extreme Degree Day (EDD) above $30^\circ\text{C}$ during anthesis.
