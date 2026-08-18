
# 📊 Customer Churn Prediction using XGBoost

An end-to-end Machine Learning project that predicts whether a bank customer is likely to churn using an **XGBoost Classifier** and an interactive **Streamlit dashboard**.

The project covers data exploration, preprocessing, categorical encoding, model training, evaluation, feature importance, model serialization, and deployment.

---

## 🚀 Live Demo

🌐 **Streamlit Application:**  
https://customer-churn-prediction-xgboost-ynnb4pspfafhcyzrewhl6k.streamlit.app/

---

## 📌 Project Overview

Customer churn is an important business problem for banks and financial institutions.

The objective of this project is to predict whether a customer is likely to leave the bank based on demographic, financial, and account-related information.

The project follows an end-to-end Machine Learning workflow:

```
Customer Dataset
       ↓
Data Exploration
       ↓
Data Preprocessing
       ↓
Categorical Encoding
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

---

## 🎯 Objective

The model performs binary classification:

| Prediction | Meaning                  |
| ---------- | ------------------------ |
| `0`        | Customer likely to stay  |
| `1`        | Customer likely to churn |

The Streamlit application accepts customer information and generates a churn prediction, probability, and risk assessment.

---

## 📊 Dataset

The project uses the **Bank Customer Churn Prediction** dataset.

### Dataset Summary

* **10,000 customer records**
* **12 original columns**
* Numerical and categorical features
* Binary `churn` target variable

The dataset contains demographic, financial, and account-related customer information.

---

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

---

## 🔍 Exploratory Data Analysis

The notebook includes exploratory analysis to understand the dataset and identify patterns associated with customer churn.

The analysis includes:

* Dataset structure
* Data types
* Missing-value analysis
* Duplicate-value analysis
* Statistical summaries
* Customer churn distribution
* Churn by country
* Age vs churn
* Active membership vs churn

---

## 🧹 Data Preprocessing

Categorical variables are converted into numerical features using Pandas `get_dummies()`.

```
ch = pd.get_dummies(
    ch,
    columns=['country', 'gender'],
    dtype=int
)
```

This creates binary features:

```
country_France
country_Germany
country_Spain
gender_Female
gender_Male
```

The customer ID and target variable are then removed from the model inputs:

```
x = ch.drop(
    columns=['customer_id', 'churn']
)

y = ch['churn']
```

The final model uses **13 input features**.

---

## ✂️ Train/Test Split

The dataset is divided into training and testing sets using an **80/20 stratified split**.

```
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=43,
    test_size=0.2,
    stratify=y
)
```

This produces:

* **8,000 training records**
* **2,000 testing records**

---

## 🤖 Machine Learning Model

### XGBoost Classifier

The project uses **XGBoost** for binary customer churn classification.

The model was trained using:

```
XGBClassifier(
    n_estimators=400
)
```

### Why XGBoost?

XGBoost is a gradient-boosting algorithm that performs well on structured and tabular datasets.

It was selected for this project because customer churn prediction is a structured classification problem involving numerical and encoded categorical features.

---

## 📈 Model Performance

The model was evaluated on the **2,000-record test set**.

| Metric                |      Score |
| --------------------- | ---------: |
| **Accuracy**          | **85.35%** |
| **Precision — Stay**  |    **89%** |
| **Recall — Stay**     |    **93%** |
| **F1-Score — Stay**   |    **91%** |
| **Precision — Churn** |    **68%** |
| **Recall — Churn**    |    **54%** |
| **F1-Score — Churn**  |    **60%** |

### Classification Report

```
              precision    recall  f1-score   support

           0       0.89      0.93      0.91      1593
           1       0.68      0.54      0.60       407

    accuracy                           0.85      2000
   macro avg       0.78      0.74      0.75      2000
weighted avg       0.84      0.85      0.85      2000
```

### Model Observation

The model achieves **85.35% accuracy**.

However, the recall for the churn class is **54%**, meaning that the model does not identify every customer who actually churns.

For a real-world churn prediction system, improving churn recall would be an important next step.

This is why accuracy alone should not be used to judge the quality of a churn prediction model.

---

## 🔍 Feature Importance

The project uses XGBoost feature importance to analyze which input features contribute most strongly to the model.

![Feature Importance](assets/feature_importance.png)

---

## 🖥️ Streamlit Application

The trained XGBoost model is integrated into an interactive Streamlit dashboard.

### Customer Inputs

Users can enter:

* Country
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
* Customer profile summary
* Key risk indicators

---

## 📸 Application Preview

![Customer Churn Intelligence Dashboard](assets/dashboard.png)

---

## 📊 Risk Assessment

The application categorizes the predicted churn probability into three risk levels:

| Churn Probability | Risk Level |
| ----------------: | ---------- |
|          0% – 30% | Low        |
|         30% – 60% | Medium     |
|        60% – 100% | High       |

These thresholds are application-level rules used to make the prediction easier to interpret.

They are not directly generated by XGBoost.

---

## 🧠 Key Risk Indicators

The dashboard provides customer-level indicators such as:

* Active membership status
* Age
* Number of products

These indicators are presented as contextual risk information.

They should not be interpreted as formal causal explanations of the individual XGBoost prediction.

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Machine Learning

* XGBoost
* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

### Model Serialization

* Pickle

### Development Tools

* Jupyter Notebook
* Google Colab
* Git
* GitHub

---

## 📁 Project Structure

```
customer-churn-prediction-xgboost/
│
├── assets/
│   ├── dashboard.png
│   └── feature_importance.png
│
├── CustomerChrun_XGBoost.ipynb
├── app.py
├── Customer-Churn-Prediction-XGBOOST.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Chaitanya-G53/customer-churn-prediction-xgboost.git
```

Navigate to the project:

```
cd customer-churn-prediction-xgboost
```

Install the required dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Start the Streamlit application:

``` 
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Machine Learning Workflow

The Colab Notebook contains the complete model development process:

1. Import libraries
2. Download the dataset
3. Load the dataset
4. Inspect dataset structure
5. Check missing values
6. Check duplicate records
7. Generate statistical summaries
8. Perform exploratory data analysis
9. Encode categorical variables
10. Separate features and target
11. Remove customer identifier
12. Split data into training and testing sets
13. Train XGBoost
14. Generate predictions
15. Generate classification report
16. Calculate accuracy
17. Analyze model performance
18. Save the trained model

---

## 💾 Model File

The trained model is stored as:

```
Customer-Churn-Prediction-XGBOOST.pkl
```

The saved file contains the trained **XGBClassifier**.

The preprocessing step is performed separately using the same categorical encoding logic used during model training.

Therefore, the input data supplied to the saved model must contain the same feature structure and column order used during training.

### Model Input Features

```
credit_score
age
tenure
balance
products_number
credit_card
active_member
estimated_salary
country_France
country_Germany
country_Spain
gender_Female
gender_Male
```

---

## ⚠️ Model Compatibility

The serialized model should be loaded using compatible Python and XGBoost environments.

When deploying the model, use the package versions specified in `requirements.txt`.

For production systems, saving the model using XGBoost's native model format can provide better portability than Python pickle serialization.

---

## 🚀 Future Improvements

### Model Improvements

* Hyperparameter optimization
* Cross-validation
* Class-imbalance handling
* Classification threshold optimization
* Improving churn recall
* Probability calibration
* Model comparison

### Explainability

* SHAP-based explanations
* Individual prediction explanations
* Global model interpretation

### Deployment

* Model monitoring
* Prediction logging
* Data drift detection
---

## 💡 Key Learning Outcomes

Through this project, I gained hands-on experience with:

* Exploratory Data Analysis
* Data preprocessing
* Categorical feature encoding
* Train/test splitting
* XGBoost classification
* Classification metrics
* Precision and recall analysis
* Feature importance
* Model serialization
* Streamlit application development
* Machine Learning deployment

---

## 📌 Key Takeaway

This project demonstrates that a Machine Learning project is more than simply training a model.

The complete workflow is:

```
Data
 ↓
Exploration
 ↓
Preprocessing
 ↓
Model Training
 ↓
Evaluation
 ↓
Prediction
 ↓
Deployment
```

The current model achieves **85.35% accuracy**, while the **54% recall for the churn class** highlights an important opportunity for future improvement.

---

## 👨‍💻 Author

### Chaitanya Girhepunje

**Aspiring Data Scientist | Python | SQL | Machine Learning**

📍 Maharashtra, India

🔗 GitHub:
[https://github.com/Chaitanya-G53](https://github.com/Chaitanya-G53)

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐.

