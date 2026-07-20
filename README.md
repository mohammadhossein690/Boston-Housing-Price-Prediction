# 🏠 Boston Housing Price Prediction using XGBoost

A machine learning project that predicts housing prices using the **XGBoost Regressor**, demonstrating hyperparameter tuning and model evaluation.

## 🎯 Objective

Predict housing prices based on property features using advanced machine learning regression techniques.

## 📈 Key Results

- **Model:** XGBoost Regressor
- **Test R² Score:** **0.8960**
- **10-Fold Cross Validation R²:** **0.8582 ± 0.0997**

### Evaluation Metrics

| Metric | Value |
|--------|------:|
| MAE | 1.8894 |
| MSE | 7.7501 |
| RMSE | 2.7839 |
| R² Score | 0.8960 |

## ⚙️ Methodology

- **Preprocessing:** Feature selection and feature scaling.
- **Model Selection:** Implementing **XGBoost Regressor** (Tree-based ensemble learning).
- **Hyperparameter Tuning:** Using **GridSearchCV** to optimize model parameters.
- **Evaluation:** Measuring model performance using R² Score, MAE, MSE, RMSE, and Cross Validation.

## 🛠 Tech Stack

Python • Pandas • NumPy • Scikit-learn • XGBoost • Matplotlib

## ▶️ How to Run

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```
