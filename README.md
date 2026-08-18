# Machine Learning Assignment 2: Classification Models & Web App Deployment

## a. Problem Statement
The objective of this project is to build, evaluate, and compare 5 distinct Machine Learning classification algorithms on tabular data. The task is to predict whether red wine is "High Quality" (Quality >= 6) vs "Low Quality" (Quality < 6) based on chemical properties, evaluate performance metrics, and deploy an interactive Streamlit application.

---

## b. Dataset Description
* **Dataset Name:** UCI Red Wine Quality Dataset
* **Source:** UCI Machine Learning Repository
* **Instance Count:** 1,599 instances
* **Feature Count:** 11 numerical input features + 1 target variable (`quality`) = 12 total attributes
* **Target Mapping:**
  * Class `1` (High Quality): Quality rating >= 6
  * Class `0` (Low Quality): Quality rating < 6

---

## c. GitHub Repository Link
**Repository URL:** https://github.com/Varad-Vanage/ml-wine-classification

---

## d. Models Used & Performance Evaluation

### Metric Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7406 | 0.8242 | 0.7683 | 0.7368 | 0.7522 | 0.4808 |
| **Decision Tree** | 0.7531 | 0.7513 | 0.7644 | 0.7778 | 0.7710 | 0.5034 |
| **KNN** | 0.7406 | 0.8117 | 0.7588 | 0.7544 | 0.7566 | 0.4790 |
| **Naive Bayes** | 0.7219 | 0.7884 | 0.7733 | 0.6784 | 0.7227 | 0.4500 |
| **Random Forest (Ensemble)** | **0.8031** | **0.9020** | **0.8293** | **0.7953** | **0.8119** | **0.6062** |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Serves as a solid linear baseline, achieving a respectable AUC of 0.8242 and balanced Precision/Recall. |
| **Decision Tree** | Captures non-linear decision thresholds well (Accuracy: 0.7531), but suffers from a lower AUC (0.7513) due to rigid binary splits. |
| **KNN** | Demonstrates consistent performance across all metrics after feature scaling, matching Logistic Regression accuracy (0.7406). |
| **Naive Bayes** | Achieves strong Precision (0.7733) but struggles with Recall (0.6784) due to independence assumptions between correlated chemical attributes. |
| **Random Forest (Ensemble)** | **Overall Winner.** Aggregates decision trees via bagging to achieve the highest performance across all evaluation metrics, notably reaching **0.8031 Accuracy**, **0.9020 AUC**, and an impressive **0.6062 MCC**. |