# Heart Disease Prediction

This project uses machine learning, specifically Logistic Regression, to predict the presence of heart disease based on a set of medical attributes. The model is trained and tested using a dataset that includes various health metrics, allowing it to distinguish between individuals with and without heart disease.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Data Exploration and Visualization](#data-exploration-and-visualization)
- [Model Development and Training](#model-development-and-training)
- [Evaluation and Results](#evaluation-and-results)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

---

## Project Overview

This project aims to build a predictive model for diagnosing heart disease based on a set of medical features. By using a Logistic Regression model, the project seeks to provide a reliable classification of heart disease presence, making it a potential tool for preliminary health assessments.

## Technologies Used

- **Python** for data processing and model training
- **Pandas** and **NumPy** for data handling and manipulation
- **Scikit-Learn** for machine learning model implementation

## Dataset

The dataset used in this project is structured in CSV format, containing various health-related attributes such as age, gender, cholesterol level, and blood pressure. The target variable (`target`) indicates whether or not heart disease is present (1 for presence, 0 for absence).

- **Dataset Link**: [Heart Disease Dataset](https://drive.google.com/drive/folders/1fttj_Xr65Puhw0IhS3tfIZTWc8-gePEC?usp=sharing)

### Data Summary
- **Rows and Columns**: The dataset consists of multiple rows and columns where each row represents a patient record.
- **Features**: Key health indicators such as age, gender, blood pressure, cholesterol, and other related metrics.
- **Target Variable**: Binary indicator (0 or 1) representing the absence or presence of heart disease.

## Data Exploration and Visualization

Basic exploratory analysis is performed to understand the data structure, check for missing values, and calculate basic statistical measures.

### Target Variable Distribution
The distribution of the target variable is analyzed to understand the proportion of patients with and without heart disease, which helps in evaluating the dataset's balance.

```python
heart_data['target'].value_counts()
```

## Model Development and Training

### Data Splitting
The dataset is split into features (`X`) and the target (`Y`). It is then divided into training and test sets with an 80-20 split, stratifying based on the target variable to maintain class balance.

### Logistic Regression Model
A Logistic Regression model is trained on the dataset to classify instances based on medical features.

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

## Evaluation and Results

The model’s performance is assessed using accuracy scores on both the training and test sets.

- **Training Data Accuracy**: Evaluates how well the model fits the training data.
- **Test Data Accuracy**: Measures the model's generalization ability on unseen data.

```python
# Accuracy on training data
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# Accuracy on test data
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the script to load, train, and evaluate the model:
   ```bash
   python heart_disease_prediction.py
   ```

2. **Sample Prediction**: Use sample data to check if an individual has heart disease. Modify `input_data` in the script for different test cases.

```python
input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
prediction = model.predict(input_data_reshaped)
```

## References

- Dataset Source: [Heart Disease Dataset on Google Drive](https://drive.google.com/drive/folders/1fttj_Xr65Puhw0IhS3tfIZTWc8-gePEC?usp=sharing)
