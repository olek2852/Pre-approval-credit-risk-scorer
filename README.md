# 🏦 Pre-approval credit risk scorer

## Overview
Data analysis and machine learning project focused on understanding and predicting preliminary credit risk. 
The system evaluates loan applications at an early stage using only data 
available before approval (such as FICO, DTI, and employment 
history), intentionally excluding post-approval metrics like loan grade or 
interest rate to simulate a real pre-approval scenario.

**[Live demo link](https://olek2852-pre-approval-credit-risk-scorer-app-k7tjc8.streamlit.app/)**

## UI Preview
<p align="left">
  <img src="form.png" width="400" height="600" alt="App Screenshot">
</p>

https://github.com/user-attachments/assets/d77b29b7-8fd5-47c5-bd6b-fa1358b06507


## Objective
In lending, reducing defaults while maintaining approval rate is critical. 
This project focuses on:
- Identifying the underlying characteristics of high-risk borrowers through extensive EDA.
- Building a robust predictive scorer to identify high-risk applicants before final approval.
- Maximizing Recall to ensure high risk borrowers are flagged to minimize potential financial losses.

## Project structure
The project is split into two stages:

**Stage 1: Data analysis & modelling (Jupyter Notebook)**  
Extensive data cleaning and exploratory data analysis on 2.2M records to extract actionable business insights. The analytical foundation was then integrated into a Scikit-Learn Pipeline for feature engineering and model training. Includes Optuna hyperparameter tuning, handling severe class imbalance (80:20), threshold optimization for Recall, and model explainability using SHAP.

**Stage 2: Interactive dashboard (Streamlit)**  
A comprehensive web app that not only provides risk scoring based on user inputs, but also allows users to explore EDA charts and SHAP values in the "About" tab to uncover the reasoning behind the data and the model's decisions.

## Model performance
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.701 |
| Gini | 0.402 |
| Recall (threshold 0.4) | 0.83 |

## Dataset
[All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) 
2.2M records, significant missing values, 80:20 class imbalance.

## Tech stack
Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, XGBoost, Optuna, Streamlit

## Repository structure
```
├── app.py                    # Streamlit app
├── credit_model.pkl          # Trained model
├── credit_notebook.ipynb     # Jupyter notebbok with EDA, preprocessing and model training
├── feature_importances.png   # SHAP feature importance plot
├── form.png                  # UI screenshot
├── requirements.txt          
├── train_small.csv           # 50K records sample dataset
```
## Run locally
1. Clone the repository
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
