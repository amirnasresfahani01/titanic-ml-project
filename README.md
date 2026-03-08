# Titanic Survival Prediction — End-to-End Data Science Project

## Project Preview

![EDA Summery](assets/Correlation%20Matrix.png)

This project demonstrates a complete end-to-end data science workflow using the Titanic dataset. The objective is to build and evaluate machine learning models that predict whether a passenger survived the Titanic disaster.

The project follows a structured machine learning pipeline including data understanding, exploratory data analysis (EDA), feature engineering, baseline modeling, model comparison, and evaluation.

---

## Key Features

- Complete end-to-end machine learning workflow
- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering including family size and passenger title
- Baseline model using Logistic Regression
- Comparison of multiple machine learning models
- Model evaluation using confusion matrix and classification report

---


# Project Workflow

The workflow of this project follows a standard data science process:

1. Data Understanding  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering  
4. Baseline Model (Logistic Regression)  
5. Model Comparison  
6. Model Evaluation  

---

# Dataset

The dataset used in this project is the Titanic dataset, which contains information about passengers such as:

- Passenger class (pclass)
- Age
- Gender
- Number of siblings/spouses aboard
- Number of parents/children aboard

The target variable is:
survived

which indicates whether a passenger survived the Titanic disaster.

---

## Project Structure
```markdown
```text
titanic-ml-project
│
├── assets
│   └── Correlation_Matrix.png
│
├── data
│   └── titanic.csv
│
├── models
│   └── titanic_model.pkl
│
├── notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   └── 05_model_comparison.ipynb
│
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── README.md
└── requirements.txt
```

# Notebooks Overview

### 01_data_understanding
This notebook performs the initial exploration of the dataset including dataset structure, missing values, and summary statistics.

### 02_eda
This notebook contains exploratory data analysis using visualizations to better understand relationships between variables and survival rates.

### 03_feature_engineering
New features are created to improve the dataset, including:

- `family_size`
- `is_alone`
- `title` extracted from passenger names

These engineered features help capture patterns that may influence survival.

### 04_baseline_model
This notebook builds a baseline machine learning model using Logistic Regression.

Steps included:
- Feature selection
- Data preprocessing
- Train-test split
- Model training
- Model evaluation

### 05_model_comparison
This notebook compares multiple machine learning models including:

- Logistic Regression
- Random Forest
- Gradient Boosting

Models are evaluated using accuracy and additional classification metrics.

---

# Machine Learning Models

The following models were tested in this project:

- Logistic Regression (baseline model)
- Random Forest
- Gradient Boosting

Each model is trained on the training dataset and evaluated on the test dataset.

---

# Model Evaluation

Model performance is evaluated using the following metrics:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

These metrics provide a more detailed understanding of model performance beyond simple accuracy.

---

# Feature Engineering

Several features were engineered to improve model performance:

- **family_size** = sibsp + parch + 1  
- **is_alone** = indicates whether the passenger was traveling alone  
- **title** extracted from passenger names  

These features help capture demographic and social patterns that influence survival probability.

## Results

| Model | Accuracy |
|------|---------|
| Logistic Regression | 0.81 |
| Random Forest | 0.79 |
| Gradient Boosting | 0.81 |

Gradient Boosting achieved the best performance in this experiment.

---

# How to Run the Project

1. Clone the repository
https://github.com/amirnasresfahani01/titanic-ml-project.git

2. Navigate to the project folder
cd titanic-ml-project

3. Install dependencies
pip install -r requirements.txt

4. Run the notebooks in order:
01_data_understanding
02_eda
03_feature_engineering
04_baseline_model
05_model_comparison

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Future Improvements

Possible improvements for this project include:

- Hyperparameter tuning
- Cross-validation
- More advanced feature engineering
- Testing additional machine learning models

---

# Author

Amir Nasr Esfahani

This project is part of my learning journey in data science and machine learning.