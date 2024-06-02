# Project Overview

    This project involves using the Python programming environment to implement various models such as Decision Tree, Logistic Regression, Random Forest, and Gradient Boosting Trees to predict 5G users. The core task is to predict whether each sample in the test set is a 5G user based on basic user information and communication-related data such as user billing information, data usage, active behavior, package type, and regional information. The evaluation metric for this task is AUC (Area Under the Curve), where a higher score indicates better performance.

# Environment Dependencies

### Python Version

    Python 3.12
### Python Packages:

    (1) pandas
    (2) sklearn
    (3) matplotlib
    (4) numpy
    (5) joblib
### IDE:

    PyCharm 2023.3.4

# Directory Structure

```
│  5G User Prediction Analysis Report.docx                // Experiment report
│  DecisionTree.py                                        // Decision Tree model script
│  LogisticRegression.py                                  // Logistic Regression model script
│  random_forest.jpynb.ipynb                              // Random Forest model script
│  xgboost.py                                             // Gradient Boosting Tree model script
│  
├─.idea                                                   // PyCharm configuration files
│  
├─.venv                                                   // Python dependencies
│  
├─data
│      train.csv                                          // Original data
│
└─jupternotebook
    │  logistic_regression.ipynb                          // IPython notebook for Logistic Regression model
    │  RandomForest.py                                    // IPython notebook for Random Forest model
    │  DecisionTree.ipynb                                 // IPython notebook for Decision Tree model
    │  xgboost.ipynb                                      // IPython notebook for Gradient Boosting Tree model
    │
└─results                                                 // Model results
    ├─decision_tree_results                               // Decision Tree results
    │      confusion_matrix_tree.png                      // Confusion matrix image
    │      decision_tree_model.pkl                        // Model file
    │      label_encoders.pkl                             // Label encoders
    │      roc_curve_tree.png                             // ROC curve image
    │      scaler.pkl                                     // Scaler model
    │      test_predictions_tree.csv                      // Test predictions
    │      
    ├─gradient_boosting_results                           // Gradient Boosting results
    │      confusion_matrix_gb.png                        // Confusion matrix image
    │      gb_model.pkl                                   // Model file
    │      roc_curve_gb.png                               // ROC curve image
    │      test_predictions_gb.csv                        // Test predictions
    │      
    ├─logistic_regression_results                         // Logistic Regression results
    │      confusion_matrix_lr.png                        // Confusion matrix image
    │      lr_model.pkl                                   // Model file
    │      roc_curve_lr.png                               // ROC curve image
    │      test_predictions_lr.csv                        // Test predictions
    │      
    ├─random_forest_results                               // Random Forest results
    │      confusion_matrix_rf.png                        // Confusion matrix image
    │      rf_model.pkl                                   // Model file
    │      roc_curve_rf.png                               // ROC curve image
    │      test_predictions_rf.csv                        // Test predictions
│  ReadMe.md                                              // Help document

```
# Authors

```
危宇豪 魏锴
```

# Version Update Summary

###### v 1.0.0
    Implemented Random Forest, Logistic Regression models.
###### v 1.1.0

```
Implemented Decision Tree model.
```

###### v 1.2.0

```
Implemented Gradient Boosting Tree model.
```

###### v 1.3.0

    Added notebook files.
###### v 1.4.0
    Added corresponding English annotations.