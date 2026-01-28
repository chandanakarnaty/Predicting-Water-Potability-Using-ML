# Predicting Water Potability Using ML

This project applies machine learning techniques to predict whether water samples are potable
based on chemical properties such as pH, hardness, sulfate, chloramines, conductivity, and turbidity.

## Dataset
- 3,276 water samples
- 9 chemical features
- Binary target: Potable / Not Potable

## Machine Learning Pipeline
- Data preprocessing (median imputation, scaling)
- Exploratory Data Analysis (EDA)
- Train-test split with stratification
- Model training and comparison

## Models Used
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Random Forest (Best-performing model)

## Results
Random Forest achieved the highest ROC-AUC (~0.68). Model performance was further improved
by optimizing the classification threshold to increase sensitivity for potable water detection.

## Tools & Libraries
R, caret, tidyverse, randomForest, pROC

