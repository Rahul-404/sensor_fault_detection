# 🚛 Air Pressure System Failure Prediction using Machine Learning

## 📌 Overview

This project aims to solve a problem where we can predict whether a truck is likely to experience a failure in the **Air Pressure System (APS)** — a critical component responsible for operations like **braking** and **gear shifting**. Early detection of potential APS failures enables proactive maintenance, reduces downtime, and ensures road safety.

---

## 📂 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks)
- **Rows:** ~36,000
- **Target Variable:** `class` (pos = APS failure, neg = non-APS failure)
- **Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Otojog7mKqfd6CoGvkQaV7eZ_Ej-kErV?usp=drive_link)

### **Features:**

- 170+ sensor readings and signal features from truck components
- All feature names are anonymized (e.g., `sensor_00`, `sensor_01`, ...)

| Feature Example  | Description                                       |
|------------------|---------------------------------------------------|
| `sensor_00`      | Sensor signal from a truck system                 |
| `sensor_01`      | not exposed dur to data privacy reasons           |
| ...              | ...                                               |
| `class`          | Target (pos = APS failure, neg = non-APS failure) |

---

## ⚙️ Project Workflow

1️⃣ **Data Preprocessing**

- Missing value imputation using constant strategy
- Encoding binary target labels
- Feature normalization and scaling
- Outlier detection and removal

2️⃣ **Exploratory Data Analysis (EDA)**

- Failure rate analysis by class
- Signal distribution comparisons
- Correlation heatmaps for feature interactions

3️⃣ **Handling Class Imbalance**

- Significant imbalance (only ~3% positive class)
- Used **SMOTETomek** and **Class Weight Balancing** to address skewed distribution

4️⃣ **Model Building**

- **Algorithms used:**
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - AdaBoost
  - XGBoost
  - CatBoost

- **Hyperparameter Tuning:** RandomizedSearchCV / GridSearchCV / Bayesian Optimization

5️⃣ **Model Evaluation**

- Metrics used: Precision, Recall, F1-Score, ROC-AUC
- **Priority on Recall & F1-Score** to reduce false positive and false negatives (critical in preventive maintenance)

6️⃣ **Model Explainability**

- Feature importance using tree-based models
- SHAP value visualization for model insights

---

## 📊 Results

| Model         | ROC-AUC | Recall | F1-Score | Total_cost |
|---------------|---------|--------|----------|------------|
| XGBoost       | 0.99    | 0.99   | 0.99     |    4050    |



---

## 🛠 Tech Stack

- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Model Explainability:** SHAP
- **API Development:** FastAPI

---

## 🚀 How to Run

```bash
git clone https://github.com/Rahul-404/sensor_fault_detection.git
cd sensor_fault_prediction
pip install -r requirements_dev.txt
python demo.py
python app.py
```

---

## 🚀 Deployment Plan

The model will be deployed as a **REST API** using **FastAPI**, hosted on **AWS EC2**, with the following architecture:

### ⚙️ **Deployment Stack**

- **FastAPI** → REST API serving stroke predictions
- **AWS S3** → Store model artifacts (e.g., trained `.pkl` files)
- **AWS ECR (Elastic Container Registry)** → Docker image registry
- **AWS EC2** → Hosting the Dockerized FastAPI service
- **Github Actions CI/CD** → Automate build, test, and deployment pipelines
- **Docker** → Containerization of the application

### ✅ **CI/CD Workflow**

1. **Push code to GitHub** → Trigger github actions Pipeline
2. **Github Actions**:

   - Build Docker image
   - Push image to **AWS ECR**
   - Deploy/update container on **AWS EC2**
3. **FastAPI** running on EC2 → Exposes `/predict` endpoint for fault detection
4. Model artifacts loaded from **AWS S3** during container startup


![Deployment-Architecture]()

---

## 📧 Contact

For questions or collaboration:

[rahulshelke3399@gmail.com](mailto:rahulshelke3399@gmail.com) |
[LinkedIn](https://www.linkedin.com/in/rahulshelke981) | [GitHub](https://github.com/Rahul-404)
