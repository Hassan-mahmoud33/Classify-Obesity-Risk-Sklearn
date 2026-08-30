# ⚖️ Multi-Class Prediction of Obesity Risk

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0.5-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.5.1-013243?logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classification-red)
![LightGBM](https://img.shields.io/badge/LightGBM-Classification-green)
![CatBoost](https://img.shields.io/badge/CatBoost-Classification-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A machine learning pipeline for predicting obesity risk level based on eating habits, physical condition, and lifestyle data. This was built for the Kaggle **Multi-Class Prediction of Obesity Risk** playground competition, and covers the full journey from raw data to a deployable web application: EDA, feature engineering, model comparison, hyperparameter tuning, a prediction API, and a simple front-end to interact with it.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [API](#api)
- [Front-End](#front-end)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

The task is a **multi-class classification** problem: given a person's age, height, weight, eating habits, and lifestyle information, predict their obesity risk category (`NObeyesdad`), ranging from insufficient weight to different obesity levels.

The project is organized into two main parts:

1. **Machine Learning pipeline** (notebooks) — from raw data to a trained, tuned model.
2. **Web application** (API + front-end) — a FastAPI backend that serves the trained model, and a simple HTML interface that lets users enter their data and get a live prediction.

## Project Structure

```
classify_obesity_risk/
│
├── api/
│   └── main.py                           # FastAPI app: loads the model and serves predictions
│
├── front_end/
│   └── index.html                        # Simple UI (Arabic/English) to submit data and view results
│
├── datasets/
│   ├── raw/                              # Original competition files
│   │   ├── sample_submission.csv
│   │   ├── test.csv
│   │   └── train.csv
│   │
│   └── processed/                        # Cleaned & feature-engineered data
│       ├── final_train.csv
│       ├── final_test.csv
│       └── submission.csv
│
├── models/                               # Trained models & encoders (saved with joblib)
│   ├── Cat_Boost_Classifier_model.pkl
│   ├── Gradient_Boosting_Classifier_model.pkl
│   ├── Random_Forest_Classifier_model.pkl
│   ├── XGB_Classifier_model.pkl
│   ├── features_encoder.pkl
│   └── target_encoder.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb                      # Exploratory data analysis
│   ├── 02_features_engineering.ipynb     # New features + encoding
│   ├── 03_train_model.ipynb              # Model comparison + tuning + evaluation
│   ├── 04_prediction.ipynb               # Generating the final submission file
│   └── catboost_info/                    # Auto-generated CatBoost training logs
│
├── run_api.bat                           # Shortcut to launch the API on Windows
└── requirements.txt
```

## Dataset

Each row represents a person, described by the following main features:

| Column | Meaning |
|---|---|
| `Age`, `Height`, `Weight` | Basic physical attributes |
| `Gender` | Male / Female |
| `family_history_with_overweight` | Whether a family member has/had overweight |
| `FAVC` | Frequent consumption of high caloric food |
| `FCVC` | Frequency of vegetable consumption |
| `NCP` | Number of main meals per day |
| `CAEC` | Consumption of food between meals |
| `SMOKE` | Whether the person smokes |
| `CH2O` | Daily water consumption |
| `SCC` | Calorie consumption monitoring |
| `FAF` | Physical activity frequency |
| `TUE` | Time spent using technology devices |
| `CALC` | Alcohol consumption |
| `MTRANS` | Transportation method used |
| `NObeyesdad` | **Target** — obesity risk category |

## Workflow

The pipeline runs through four notebooks, in order:

1. **`01_EDA.ipynb`** – Exploring the data: checking for duplicates and missing values, and reviewing the distribution of every feature (age, height, weight, eating habits, lifestyle columns, and the target classes).
2. **`02_features_engineering.ipynb`** – Creating new features, encoding all categorical columns, and saving the fitted encoders for later use during prediction.
3. **`03_train_model.ipynb`** – Comparing multiple classification models with cross-validation, tuning the top candidates, and evaluating the final chosen model.
4. **`04_prediction.ipynb`** – Loading the best model and encoders, predicting on the test set, and building the final `submission.csv` file.

Once the model is trained, the same encoders and model are reused by the **API** to serve live predictions through the front-end.

## Feature Engineering

New features were derived to better capture the relationship between body measurements and lifestyle habits:

| Feature | Description |
|---|---|
| `BMI` | Body Mass Index (Weight / Height²) — the strongest predictor of obesity class |
| `Weight_Height_Ratio` | Linear ratio of weight to height, complementing BMI |
| `Ideal_Weight` | Estimated ideal weight (Devine formula approximation) |
| `Weight_Deviation` | Difference between actual weight and estimated ideal weight |
| `FCVC_NCP_Ratio` | Ratio of vegetable consumption frequency to number of meals |
| `FAF_TUE_Ratio` | Ratio of physical activity frequency to technology usage time |
| `Age_Group` | Age bucketed into Teen, Young Adult, Adult, Middle Age, and Senior |
| `Active_Transport` | Whether the person uses active transportation (walking or biking) |

All categorical columns (including the target) were label-encoded, with the fitted encoders saved separately (`features_encoder.pkl`, `target_encoder.pkl`) so the exact same encoding can be reused during inference. The API rebuilds these exact same features at prediction time (see [API](#api) below), ensuring consistency between training and serving.

## Models

Six classification models were first compared using repeated stratified k-fold cross-validation (accuracy, F1-macro, precision, and recall):

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGB Classifier
- CatBoost Classifier

The four best performers (**CatBoost**, **XGB**, **Gradient Boosting**, and **Random Forest**) were then fine-tuned using `RandomizedSearchCV`, and each tuned model was saved to the `models/` folder.

📈 **Final model used for predictions:** `XGB_Classifier`, evaluated with a confusion matrix, accuracy, macro-averaged ROC-AUC, and a full classification report.

## API

The `api/main.py` file exposes the trained model as a **FastAPI** REST service.

### What it does

1. **Loads artifacts on startup** — the tuned `XGB_Classifier_model.pkl`, along with the saved `features_encoder.pkl` and `target_encoder.pkl`, so predictions are encoded and decoded exactly the way they were during training.
2. **Validates incoming data** — a `PatientData` Pydantic model defines and validates the expected input schema (gender, age, height, weight, eating habits, lifestyle fields, etc.), rejecting malformed requests automatically.
3. **Rebuilds the engineered features** — the `build_features()` function reproduces the same feature engineering used in the notebooks (`BMI`, `Weight_Height_Ratio`, `Ideal_Weight`, `Weight_Deviation`, `FCVC_NCP_Ratio`, `FAF_TUE_Ratio`, `Age_Group`, `Active_Transport`), then encodes the categorical columns with the saved encoders before feeding the data into the model. If a category value wasn't seen during training, the API returns a clear `400` error instead of crashing.
4. **Serves the front-end** — the app mounts the `front_end/` folder as static files at the root path (`/`), so opening the API's URL in a browser loads `index.html` directly. No separate web server is needed.

### Endpoint

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Accepts a JSON body matching `PatientData` and returns the predicted class with a confidence score. |

**Example request body:**

```json
{
  "Gender": "Male",
  "Age": 25,
  "Height": 1.75,
  "Weight": 80,
  "family_history_with_overweight": "yes",
  "FAVC": "yes",
  "FCVC": 2,
  "NCP": 3,
  "CAEC": "Sometimes",
  "SMOKE": "no",
  "CH2O": 2,
  "SCC": "no",
  "FAF": 1,
  "TUE": 1,
  "CALC": "no",
  "MTRANS": "Public_Transportation"
}
```

**Example response:**

```json
{
  "prediction": "Overweight_Level_I",
  "confidence": 0.87
}
```

CORS is enabled so the front-end (or any external client) can call the API without cross-origin issues.

## Front-End

`front_end/index.html` is a lightweight, dependency-free interface (Arabic UI, right-to-left layout) that lets a user fill in their information and get an instant prediction — no build tools or frameworks required, just plain HTML, CSS, and vanilla JavaScript.

- The form collects all fields required by the API and sends them as JSON to the `/predict` endpoint.
- It automatically detects whether it's being opened as a local file or served by the API, and adjusts the request URL accordingly (`http://127.0.0.1:8000/predict` locally, or `/predict` when served by FastAPI).
- The predicted class and confidence score are displayed directly on the page, with both English and Arabic labels for the result, and a clear error message if the request fails.

Since the API serves this file directly, you don't need to open `index.html` separately — just run the API and visit it in your browser.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/<your-username>/classify_obesity_risk.git
cd classify_obesity_risk
pip install -r requirements.txt
```

**requirements.txt**
```
pandas==3.0.5
numpy==2.5.1
matplotlib==3.11.1
seaborn
scikit-learn
xgboost
lightgbm
catboost
joblib
jupyter
fastapi
uvicorn
```

## How to Run

### 1. Run the ML pipeline (optional — models are already trained and saved)

The notebooks were developed and tested in **VS Code** using the Python and Jupyter extensions, but any Jupyter environment works the same way (JupyterLab, classic Jupyter Notebook, or Kaggle itself).

Run the notebooks in order from the `notebooks/` folder:

```
01_EDA.ipynb   →   02_features_engineering.ipynb   →   03_train_model.ipynb   →   04_prediction.ipynb
```

Each notebook reads its input from `datasets/raw/` or `datasets/processed/` (depending on the stage) and writes its output back to `datasets/processed/`. The final submission file will be generated at `datasets/processed/submission.csv`, ready to upload to Kaggle.

> **Note:** The code automatically detects whether it's running inside a Kaggle notebook (`/kaggle/input`) or locally, and adjusts the data paths accordingly — no manual path changes needed.

### 2. Run the web app (API + front-end)

From the project root, start the API with `uvicorn`:

```bash
uvicorn api.main:app --reload
```

Or, on Windows, simply double-click `run_api.bat`.

Once it's running, open your browser at:

```
http://127.0.0.1:8000
```

This loads the front-end directly (served by the API), where you can fill in the form and get a live prediction. Interactive API docs are also available automatically at:

```
http://127.0.0.1:8000/docs
```

## Tech Stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **Model persistence:** Joblib
- **API:** FastAPI, Uvicorn
- **Front-end:** HTML, CSS, vanilla JavaScript

## License

This project is licensed under the MIT License.
