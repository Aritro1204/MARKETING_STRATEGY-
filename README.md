# 🧳 Holiday Package Purchase Prediction

## 📌 Project Overview
This project aims to build a machine learning model that predicts whether a customer is likely to purchase a holiday package. It is designed for **Trips & Travel.com**, which is planning to introduce a new **Wellness Tourism Package** and wants to use data-driven methods to optimize its marketing strategy and reduce costs.

---

## 💡 Problem Statement

Currently, only ~18% of contacted customers end up purchasing a package, yet **marketing costs are high** because outreach is random. The objective is to:
- Use customer data to build a predictive model.
- Identify customers most likely to purchase a holiday package.
- Reduce unnecessary marketing spending.

---

## 📊 Dataset

- 📁 Source: [Kaggle Dataset](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)
- 🔢 Rows: 4,888
- 🔣 Columns: 20
- 🎯 Target Variable: `ProdTaken` (1 = Package Purchased, 0 = Not Purchased)

---

## 🛠️ Tools & Technologies

- **Programming Language:** Python
- **Libraries:**
  - `pandas`, `numpy` — Data manipulation
  - `matplotlib`, `seaborn`, `plotly` — Data visualization
  - `scikit-learn` — Preprocessing, Modeling, Evaluation
- **Model Types:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Tuning:** `RandomizedSearchCV`

---

## 🔍 Project Workflow

### 1. 📥 Data Loading & Cleaning
- Loaded CSV data using `pandas`.
- Cleaned anomalies (`"Fe Male"` → `"Female"`, `"Single"` → `"Unmarried"`).
- Handled missing values with:
  - Median for numerical columns.
  - Mode for categorical and discrete integer columns.
- Removed irrelevant features like `CustomerID`.

### 2. ⚙️ Feature Engineering
- Created `TotalVisiting = NumberOfPersonVisiting + NumberOfChildrenVisiting`.
- Dropped the original two columns.

### 3. 📊 Feature Categorization
- Identified:
  - `Numerical Features` (int/float)
  - `Categorical Features` (object)
  - `Discrete` vs `Continuous` numerical features

### 4. 🔁 Train-Test Split
- Split the data into:
  - `X_train`, `X_test`
  - `y_train`, `y_test`
- Used `train_test_split(test_size=0.2, random_state=42)`

### 5. 🔄 Preprocessing with ColumnTransformer
- Applied:
  - `StandardScaler` on numerical features
  - `OneHotEncoder` (drop='first') on categorical features

### 6. 🤖 Model Training
- Trained and evaluated:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Evaluated using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **ROC-AUC Score**

### 7. 🔧 Hyperparameter Tuning
- Used `RandomizedSearchCV` to tune the **Random Forest Classifier** over:
  - `n_estimators`, `max_depth`, `min_samples_split`, `max_features`

### 8. 📈 Model Evaluation & Visualization
- Evaluated tuned Random Forest model.
- Plotted **ROC-AUC Curve** to visualize classifier performance.

---

## 📈 Final Model Performance (Random Forest)

| Metric         | Train Set | Test Set |
|----------------|-----------|----------|
| Accuracy       | ~0.88     | ~0.83    |
| Precision      | ~0.86     | ~0.81    |
| Recall         | ~0.80     | ~0.76    |
| F1 Score       | ~0.83     | ~0.78    |
| ROC AUC Score  | ~0.91     | ~0.83    |

---

## 📊 ROC-AUC Curve

![ROC Curve](auc.png)

---

## 📂 Folder Structure

