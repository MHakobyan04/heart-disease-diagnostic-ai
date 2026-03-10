# Model Performance Report
### Executive Summary
The goal of this project was to develop a robust binary classifier to identify heart disease risk.
After benchmarking multiple algorithms, Logistic Regression was optimized using GridSearchCV to
maximize Recall, ensuring minimal false negatives in a clinical context.
### Final Metrics
The following results were achieved on the unseen test set:
 Metric    Value   Rationale
Accuracy   80.33%  Overall correct predictions across both classes. 
Recall     85.00%  Critical for medical safety; minimizes missed cases.
ROC-AUC    0.871   High discriminative power between classes.  
### Hyperparameter Optimization
To prevent overfitting and ensure stability, we performed a grid search over the following space:___
* Regularization (C): [0.1, 1, 10, 100] 
* Penalty: L2 (Ridge)
* Validation Strategy: 5-Fold Stratified Cross-Validation
### Feature Importance
The model identified the following features as the strongest predictors:
* Chest Pain Type (cp): Higher values strongly correlate with disease presence.
* Max Heart Rate (thalach): A significant physiological indicator.
* ST Depression (oldpeak): Indicates cardiac stress during exercise.
