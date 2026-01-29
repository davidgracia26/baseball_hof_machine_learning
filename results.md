Using all columns for hitters, this is what the hyperparameters were for xgboost
Best Hyperparameters (GridSearchCV): {'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.01, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 100, 'objective': 'binary:logistic', 'reg_alpha': 0.1, 'reg_lambda': 1.5, 'scale_pos_weight': 1, 'subsample': 0.8}
Best Score (GridSearchCV): 0.96
Test Precision of the Best XGBoost Model: 1.0

Classification Report of the Best XGBoost Model:
              precision    recall  f1-score   support

         0.0       0.99      1.00      1.00      3724
         1.0       1.00      0.26      0.41        46

    accuracy                           0.99      3770
   macro avg       1.00      0.63      0.70      3770
weighted avg       0.99      0.99      0.99      3770

Using all columns for hitters, this is what the hyperparameters were for xgboost
Best Hyperparameters (RandomizedSearchCV): {'subsample': 0.8, 'scale_pos_weight': 1, 'reg_lambda': 1.5, 'reg_alpha': 0.1, 'objective': 'binary:logistic', 'n_estimators': 100, 'min_child_weight': 1, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0.1, 'colsample_bytree': 1.0}
Best Score (RandomizedSearchCV): 0.7357902097902098
Test Precision of the Best XGBoost Model: 0.71875

Classification Report of the Best XGBoost Model:
              precision    recall  f1-score   support

         0.0       0.99      1.00      1.00      3724
         1.0       0.72      0.50      0.59        46

    accuracy                           0.99      3770
   macro avg       0.86      0.75      0.79      3770
weighted avg       0.99      0.99      0.99      3770

Using VIF 1.99 threshold, this is what the hyperparameters were for xgboost
Best Hyperparameters (GridSearchCV): {'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 200, 'objective': 'binary:logistic', 'reg_alpha': 0, 'reg_lambda': 1.5, 'scale_pos_weight': 1, 'subsample': 0.8}
Best Score (GridSearchCV): 0.9199999999999999
Test Precision of the Best XGBoost Model: 0.8333333333333334

Classification Report of the Best XGBoost Model:
              precision    recall  f1-score   support

         0.0       0.99      1.00      0.99      3724
         1.0       0.83      0.11      0.19        46

    accuracy                           0.99      3770
   macro avg       0.91      0.55      0.59      3770
weighted avg       0.99      0.99      0.98      3770

Using VIF 1.99 threshold, this is what the hyperparameters were for xgboost
Best Hyperparameters (RandomizedSearchCV): {'subsample': 1.0, 'scale_pos_weight': 1, 'reg_lambda': 1.5, 'reg_alpha': 0, 'objective': 'binary:logistic', 'n_estimators': 100, 'min_child_weight': 5, 'max_depth': 3, 'learning_rate': 0.05, 'gamma': 0.2, 'colsample_bytree': 1.0}
Best Score (RandomizedSearchCV): 0.7837662337662338
Test Precision of the Best XGBoost Model: 0.7142857142857143

Classification Report of the Best XGBoost Model:
              precision    recall  f1-score   support

         0.0       0.99      1.00      0.99      3724
         1.0       0.71      0.22      0.33        46

    accuracy                           0.99      3770
   macro avg       0.85      0.61      0.66      3770
weighted avg       0.99      0.99      0.99      3770