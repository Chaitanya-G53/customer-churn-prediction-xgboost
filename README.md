# 📊 Customer Churn Prediction using XGBoost

An end-to-end Machine Learning project that predicts whether a bank customer is likely to churn using an **XGBoost classification model**.

The project covers the complete Machine Learning workflow — from exploratory data analysis and preprocessing to model evaluation, feature importance, model persistence, and deployment through an interactive **Streamlit dashboard**.

## 🚀 Live Demo

🌐 **Streamlit Application:**  
https://customer-churn-prediction-xgboost-ynnb4pspfafhcyzrewhl6k.streamlit.app/

## 📌 Project Overview

Customer churn is an important business problem for banks and financial institutions.

The objective of this project is to predict whether a customer is likely to leave the bank based on demographic, financial, and account-related information.

The trained XGBoost model is combined with a Scikit-learn preprocessing pipeline and deployed as an interactive Streamlit application.

### End-to-End Workflow

```
Customer Dataset
       ↓
Exploratory Data Analysis
       ↓
Data Preprocessing
       ↓
One-Hot Encoding
       ↓
Train/Test Split
       ↓
XGBoost Classifier
       ↓
Model Evaluation
       ↓
Feature Importance
       ↓
Model Serialization
       ↓
Streamlit Deployment
       ↓
Churn Prediction
````
## 🎯 Objective

The model performs binary classification:

| Prediction | Meaning                  |
| ---------- | ------------------------ |
| `0`        | Customer likely to stay  |
| `1`        | Customer likely to churn |

The application also calculates the predicted churn probability and displays a corresponding risk level.

## 📊 Dataset

The project uses a bank customer churn dataset containing:

* **10,000 customer records**
* **12 columns**
* Numerical and categorical features
* `churn` as the target variable

The dataset contains customer demographic, financial, and account-related information.

## 🧾 Features

| Feature            | Description                              |
| ------------------ | ---------------------------------------- |
| `customer_id`      | Unique customer identifier               |
| `credit_score`     | Customer credit score                    |
| `country`          | Customer's country                       |
| `gender`           | Customer gender                          |
| `age`              | Customer age                             |
| `tenure`           | Number of years with the bank            |
| `balance`          | Customer account balance                 |
| `products_number`  | Number of products used by the customer  |
| `credit_card`      | Whether the customer has a credit card   |
| `active_member`    | Whether the customer is an active member |
| `estimated_salary` | Estimated customer salary                |
| `churn`            | Target variable                          |

`customer_id` is excluded from model training because it is an identifier rather than a predictive feature.

## 🔍 Exploratory Data Analysis

The notebook performs exploratory analysis to understand the dataset and identify patterns related to customer churn.

The analysis includes:

* Dataset structure and data types
* Missing-value analysis
* Duplicate-value analysis
* Statistical summaries
* Churn distribution
* Country-wise churn analysis
* Age distribution
* Active membership and churn analysis
* Customer feature relationships

## 🧹 Data Preprocessing

The project uses a Scikit-learn preprocessing pipeline.

Categorical variables such as:

* `country`
* `gender`

are transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

A `ColumnTransformer` is used to apply the appropriate preprocessing to categorical features while passing numerical features through.

The preprocessing stage and XGBoost model are combined into a single Scikit-learn `Pipeline`.

```text
Input Data
    ↓
ColumnTransformer
    ↓
OneHotEncoder
    ↓
XGBoost Classifier
    ↓
Prediction
```

This ensures that the same preprocessing logic is automatically applied during both training and prediction.

## 🤖 Machine Learning Model

### XGBoost Classifier

The project uses **XGBoost** for binary customer churn classification.

Current model configuration:

```python
XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    eval_metric="logloss"
)
```

### Why XGBoost?

XGBoost was selected because it is a powerful gradient-boosting algorithm that performs well on structured/tabular datasets and classification problems.

## 📈 Model Performance

The model was evaluated on a held-out test set.

| Metric                |      Score |
| --------------------- | ---------: |
| **Accuracy**          | **86.55%** |
| **ROC-AUC**           | **86.41%** |
| **Precision — Churn** |    **78%** |
| **Recall — Churn**    |    **48%** |
| **F1-Score — Churn**  |    **59%** |

### Classification Report

```
              precision    recall  f1-score

0                0.88       0.96      0.92
1                0.78       0.48      0.59

Accuracy                         0.87
```

### Important Observation

The model achieves good overall classification performance and an ROC-AUC of **0.8641**.

However, the recall for the churn class is **48%**, meaning that a significant number of actual churn customers are still missed.

This highlights an important limitation of the current model and provides a clear direction for future improvement.

## 🔲 Confusion Matrix

The final model produced the following confusion matrix:

```
                 Predicted
                 Stay   Churn

Actual Stay      1537    56
Actual Churn      213   194
```

### Interpretation

* **1537** customers were correctly predicted as staying.
* **56** customers were incorrectly classified as churn.
* **213** actual churn customers were missed by the model.
* **194** churn customers were correctly identified.

The false negatives are particularly important in a churn prediction problem because these represent customers who actually churned but were not identified by the model.

![Confusion Matrix](assets/confusion_matrix.png)

## 🔍 Feature Importance

XGBoost feature importance is extracted from the trained model after preprocessing.

The feature-importance analysis helps identify which transformed features contribute most to the model.

![Feature Importance](assets/feature_importance.png)

## 🖥️ Streamlit Application

The trained Machine Learning pipeline is deployed through an interactive Streamlit dashboard called:

### Customer Churn Intelligence Dashboard

The application allows users to enter customer information and receive a churn prediction.

### Customer Inputs

* Country / Geography
* Gender
* Age
* Credit Score
* Tenure
* Account Balance
* Number of Products
* Credit Card Status
* Active Membership
* Estimated Salary

### Application Output

The dashboard displays:

* Customer churn prediction
* Retention probability
* Churn probability
* Risk level
* Probability gauge
* Customer feature profile
* Model information

## 📸 Application Preview

![Customer Churn Dashboard](assets/dashboard.png)

## 📊 Dashboard Features

### Model Engine

```
XGBoost Classifier
```

### Model Performance

```
ROC-AUC: 86.41%
```

### Decision Threshold

```
50%
```

### Prediction Overview

The application displays:

* Likely to Stay / Likely to Churn
* Retention Probability
* Churn Probability
* Risk Level

### Probability Gauge

A visual gauge represents the predicted churn probability.

### Customer Feature Profile

The application displays the feature values submitted to the ML pipeline.

## ⚠️ Risk Level

The dashboard categorizes churn probability into three levels:

| Churn Probability | Risk Level |
| ----------------: | ---------- |
|          0% – 30% | Low        |
|         30% – 60% | Medium     |
|        60% – 100% | High       |

These risk categories are application-level thresholds used to make the model output easier to interpret.

They are **not directly produced by XGBoost**.


## 🛠️ Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

### Model Persistence

* Pickle

### Development

* Jupyter Notebook
* Google Colab
* Git
* GitHub

## 📁 Project Structure

```
customer-churn-prediction-xgboost/
│
├── assets/
│   ├── dashboard.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── Customer_Churn.ipynb
├── app.py
├── customer_churn_xgboost_pipeline.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction-xgboost.git
```

Navigate to the project directory:

```
cd customer-churn-prediction-xgboost
```

Install the dependencies:

```
pip install -r requirements.txt
```

## ▶️ Run Locally

Start the Streamlit application:

```
streamlit run app.py
```

The application will open in your browser.

## 🧪 Model Development Workflow

The Colab Notebook contains the complete model development process:

1. Import required libraries
2. Load the dataset
3. Inspect dataset structure
4. Check data types
5. Check missing values
6. Check duplicate records
7. Perform exploratory data analysis
8. Analyze churn distribution
9. Analyze customer characteristics
10. Separate features and target
11. Remove customer identifier
12. Split data into training and testing sets
13. Apply categorical preprocessing
14. Train XGBoost
15. Generate predictions
16. Generate prediction probabilities
17. Evaluate classification performance
18. Generate confusion matrix
19. Calculate ROC-AUC
20. Analyze feature importance
21. Save the trained pipeline

## 💾 Model Persistence

The complete preprocessing and model pipeline is saved as:

```
customer_churn_xgboost_pipeline.pkl
```

Saving the complete pipeline ensures that preprocessing and prediction use the same transformations.

The deployment environment should use compatible versions of the libraries used when the model was trained.

## 🚀 Future Improvements

### Model Improvements

* Hyperparameter optimization
* Cross-validation
* Class-imbalance handling
* Classification threshold optimization
* Probability calibration
* Improving churn recall

* SHAP-based local explanations
* Global feature importance visualization
* Individual prediction explanations

### Deployment

* Model monitoring
* Prediction logging
* Data drift detection
* Automated model retraining


### Explainability
### Model Comparison

Compare XGBoost against:

* Logistic Regression
* Random Forest
* AdaBoost
* Other gradient-boosting models

## 💡 Key Learning Outcomes

This project provided hands-on experience with an end-to-end Machine Learning workflow:

* Exploratory Data Analysis
* Data cleaning and validation
* Feature preprocessing
* One-Hot Encoding
* ColumnTransformer
* Scikit-learn Pipelines
* XGBoost classification
* Classification metrics
* Confusion matrix
* ROC-AUC
* Feature importance
* Model serialization
* Streamlit application development
* ML model deployment

## 📌 Key Takeaway

Building a Machine Learning model is only one part of an ML workflow.

A complete ML application also requires:

```
Data
 ↓
Preprocessing
 ↓
Model
 ↓
Evaluation
 ↓
Interpretation
 ↓
Deployment
```

The current model provides a strong baseline with **86.55% accuracy and 86.41% ROC-AUC**, while the lower churn recall highlights a real area for further model improvement.

## 👨‍💻 Author

### Chaitanya Girhepunje

**Aspiring Data Scientist | Python | SQL | Machine Learning**

📍 Bhandara, Maharashtra

🔗 GitHub: [Chaitanya-G53](https://github.com/Chaitanya-G53)


## ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐.

