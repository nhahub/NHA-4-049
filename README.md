# Predictive Maintenance System

A machine learning system that predicts whether an industrial machine is likely to require maintenance within 7 days, based on sensor data.

---

## Project Overview

This project uses a classification model trained on IoT sensor data from 500,000 industrial machine, The goal is to shift from reactive maintenance to proactive, data-driven scheduling — reducing unexpected downtime & optimizing maintenance planning.

---

## Project Structure

```text
Machine_tracking_Prediction/
├── data/
│   ├── raw/
│   │   └── factory_sensor.csv  
│   └── processed/
│       └── factory_sensor_processed.csv
│
├── machine_break_prediction/
│   ├── 01_eda.ipynb                 
│   └── 02_model.ipynb            
│
├── app/
│   └── app.py
│                                     
├── models/
│   └── model.pkl
│
├── docs/
│   └── project_notes.md   
│
├── model_columns.pkl                
├── requirements.txt                  
└── README.md (you are here)
```
---

## Dataset

- **Source:** Synthetic Industrial IoT factory sensor dataset
- **Size:** 500,000 machines
- **Target:** `Failure_Within_7_Days` (binary: 0 = Safe, 1 = Needs Maintenance)

**Input Features:**

| Feature | Description |
|---|---|
| Temperature_C | Machine operating temperature |
| Vibration_mms | Vibration level in mm/s |
| Sound_dB | Sound level in decibels |
| Oil_Level_pct | Oil level percentage |
| Coolant_Level_pct | Coolant level percentage |
| Coolant_Flow_L_min | Coolant flow rate |
| Power_Consumption_kW | Power consumption |
| Heat_Index | Combined heat and humidity measure |
| Hydraulic_Pressure_bar | Hydraulic pressure (press machines) |
| Laser_Intensity | Laser intensity (laser cutters) |
| Last_Maintenance_Days_Ago | Days since last maintenance |
| Maintenance_History_Count | Total maintenance count |
| Failure_History_Count | Total past failures |
| Error_Codes_Last_30_Days | Error codes in last 30 days |
| Machine_Type | Type of industrial machine (one-hot encoded) |

---

## Key Findings

- `Remaining_Useful_Life_days` and `Operational_Hours` were identified as **data leakage**
- After removing leakage, legitimate features show low linear correlation with the target — consistent with a synthetically generated dataset
- Dataset is heavily imbalanced (~94% safe, ~6% failing) — handled using `OverSampling(Smote)`

---

## Installation

```bash
git clone https://github.com/Candy-Slayer/Machine_tracking_Prediction.git
cd Machine_tracking_Prediction
pip install -r requirements.txt
```

---

**Run the notebooks in order:**
1. `machine_break_prediction/01_eda.ipynb`
2. `machine_break_prediction/02_model.ipynb`

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python | language |
| Pandas | Data manipulation |
| Scikit-learn | Model training |
| Imbalanced-learn | Handling class imbalance |
| Matplotlib + Seaborn | Visualization |
| Streamlit | Web dashboard |
| Joblib | Model serialization |
| GitHub | Version control |

---

## Model

- **Algorithm:** Random Forest Classifier + XGBOOST Classifier
- **Evaluation Metrics:** Precision, Recall, F1-Score, Confusion Matrix
- **Primary Metric:** Recall (class 1) — missing a real failure is costlier than a false alarm

---

##  How to Run

**Run the Streamlit dashboard:**
```bash
streamlit run app.py
```

## Notes (only read if you want to train model on your local device, or experiment on it)
- **add "data" folder**
- **add 2 sub-folders (raw/processed)**
- **download data from https://www.kaggle.com/datasets/canozensoy/industrial-iot-dataset-synthetic**
- **add it to raw folder (rename to factory_sensor)**
- **run 01_eda file**
- **run 02_training file**
