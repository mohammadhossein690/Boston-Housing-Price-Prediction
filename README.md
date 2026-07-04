# Boston Housing Price Prediction

A machine learning project that predicts housing prices using the **XGBoost Regressor**, demonstrating hyperparameter tuning and model evaluation.

## Key Results
- **Model:** XGBoost Regressor
- **Best Parameters:** `{'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 600, 'subsample': 0.7}`
- **10-Fold Cross-Validation (R²):** 0.858
- **Test Set Performance (R²):** 0.896

## Methodology
- Preprocessing: Feature selection and scaling.
- Model Selection: XGBoost (Tree-based ensemble).
- Hyperparameter Tuning: GridSearchCV to optimize model parameters.

## How to Run
1. Install dependencies:
```bash
   pip install numpy pandas matplotlib scikit-learn xgboost
   
