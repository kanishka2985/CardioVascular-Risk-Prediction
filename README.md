# ❤️🩺 Cardiovascular Risk Prediction Model 📊⚕️

## 📌 Project Overview
This project aims to **predict the 10‑year cardiovascular disease (CVD) risk** for individuals using machine learning techniques.  
Accurate risk prediction can assist clinicians and public health professionals in early interventions, lifestyle recommendations, and treatment planning.

---

## 🔍 Problem Statement
Cardiovascular diseases are among the leading causes of mortality worldwide.  
Identifying high‑risk individuals early is crucial to reduce morbidity and improve outcomes.  
This project builds a predictive model using clinical and lifestyle data to estimate the probability of developing cardiovascular disease within 10 years.

---

## 📂 Dataset
**Source:** [Cardiovascular Risk Prediction](https://www.kaggle.com/code/bansodesandeep/cardiovascular-risk-prediction)  
**Size:** ~4,000 records with patient features.

**Key Features:**
- `age`
- `sex`
- `education`
- `cigsPerDay` (cigarettes per day)
- `BPMeds` (blood pressure medication usage)
- `prevalentStroke`
- `prevalentHyp` (hypertension)
- `diabetes`
- `totChol` (total cholesterol)
- `sysBP`, `diaBP` (blood pressure)
- `BMI`
- `heartRate`
- `glucose`
- **Target:** `TenYearCHD` (1 = develops CHD, 0 = does not)

---

## ⚙️ Technologies Used
- Python 🐍  
- Pandas & NumPy  
- Scikit‑learn  
- Matplotlib & Seaborn  
- Jupyter Notebook  

---

## 🧪 Methodology

### 🔹 Data Preprocessing
✔️ Handled missing values and outliers  
✔️ Engineered additional features (e.g., `pulse_pressure`, `is_obese`, `smoking_intensity`, `glucose_status`)  
✔️ Normalized and scaled continuous variables  

### 🔹 Exploratory Data Analysis (EDA)
✔️ Visualized distributions and correlations  
✔️ Generated boxplots to detect outliers  
✔️ Examined relationships between risk factors and `TenYearCHD`

### 🔹 Modeling Techniques
Tested and compared multiple classification models:
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- XGBoost Classifier

Each model was evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1‑Score
- ROC‑AUC

---

## 🧠 Models Trained

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| Decision Tree Classifier | Tree-based classification model |
| XGBoost | Advanced gradient boosting model |

---

## 📈 Results

| Model | Precision | Recall | F1‑Score | ROC‑AUC |
|-------|-----------|--------|----------|----------|
| Logistic Regression (Default) | 0.804476 | 0.727370 | 0.763982 | 0.775293 |
| Logistic Regression (Tuned) | 0.803066 | 0.725240 | 0.762171 | 0.773695 |
| Random Forest (Default) | 0.840173 | 0.828541 | 0.834316 | 0.835463 |
| Random Forest (Tuned) | 0.871508 | 0.830671 | 0.850600 | 0.854100 |
| Decision Tree (Default) | 0.787559 | 0.714590 | 0.749302 | 0.760916 |
| Decision Tree (Tuned) | 0.832765 | 0.779553 | 0.805281 | 0.811502 |
| XGBoost (Default) | **0.909091** | **0.862620** | **0.885246** | **0.888179** |
| XGBoost (Tuned) | 0.948655 | 0.826411 | 0.883324 | 0.890841 |

✅ **Best Performing Model:**  
**XGBoost (Default)** achieved the best overall performance with:
- **Precision:** 0.9091  
- **Recall:** 0.8626  
- **ROC‑AUC:** 0.8882  

---

## ▶️ How to Run

Clone this repository:
```bash
git clone https://github.com/kanishka2985/cardiovascular-risk-prediction.git
cd cardiovascular-risk-prediction
