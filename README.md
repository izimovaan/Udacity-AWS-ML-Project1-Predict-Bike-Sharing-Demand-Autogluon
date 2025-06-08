# Predict Bike Sharing Demand with AutoGluon

## Project Overview
This project uses AutoGluon to predict bike sharing demand based on the Kaggle Bike Sharing Demand dataset. The main focus was on feature engineering and model tuning to improve prediction accuracy.

Starting from baseline features and default settings, the model’s RMSE improved significantly through careful feature creation and preprocessing. Hyperparameter tuning provided limited gains compared to the impact of feature engineering.

## Key Features
- Datetime decomposition (year, month, day of week, hour)
- Categorical encoding of season and weather
- Removal of redundant and leakage-prone variables
- Addition of engineered features like temperature categories and activity levels

## Results
The best model achieved a substantial reduction in RMSE, demonstrating the importance of domain-informed feature design combined with AutoGluon’s automated modeling capabilities.

## Dependencies
- Python 3.11.13
- AutoGluon 1.3.1  
- MXNet 1.9.1  
- Bokeh 3.7.3  
