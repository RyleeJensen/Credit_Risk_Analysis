# Credit_Risk_Analysis

## Overview
The purpose of this project was to analyze credit card risk. This particular issue is an inherently unbalanced classification problem as good loans easily outnumber risky loans. We will use a handful of different machine learning models to see which one is best a prediction credit risk. These include:
  - Oversampling the data using the RandomOverSampler and SMOTE algorithms
  - Undersampling the data using the ClusterCentroids algorithm
  - Use a combination approach of over and undersampling the data with the SMOTEENN algorithm
  - Compare two machine learning models: BalancedRandomForestClassifier and EasyEnsembleClassifier

## Resources
- Data Source: LoanStats_2019Q1.csv
- Tools: Python 3.8.8, scikit-learn, Pandas, Numpy, Jupyter Notebook

## Results
### RandomOverSampler
![ROS](https://github.com/RyleeJensen/Credit_Risk_Analysis/blob/main/Images/RandomOverSampler.png)

The balanced accuracy for the RandomOverSampler came out to 65%. We see the high risk precision is only about 0.01 and the sensitivity (rec) is 0.62. This makes for a F1 score (or harmonic mean) of .02 (A low F1 score shows a pronounced imbalance between precision and sensitivity. The low risk precision is 1.00, the sensitivity is 0.68, and the F1 score is 0.81.

### SMOTE
![SMOTE](https://github.com/RyleeJensen/Credit_Risk_Analysis/blob/main/Images/SMOTE.png)

The results of the SMOTE algorithm is similar to that of the RandomOverSampler algorithm. The high risk precision is 0.01, the sensitivity is 0.62, and the F1 score is 0.02. The low risk precision is 1.00, the sensitivity is 0.68, and the F1 score is 0.81.

### ClusterCentroids 
![ClusterCentroids](https://github.com/RyleeJensen/Credit_Risk_Analysis/blob/main/Images/ClusterCentroids.png)

The high risk precision is 0.01, the sensitivity 0.61, and the F1 score is 0.01. The low risk precision is 1.00, the sensitivity is 0.45, and the F1 score is 0.62.

### SMOTEENN
![SMOTEENN](https://github.com/RyleeJensen/Credit_Risk_Analysis/blob/main/Images/SMOTEENN.png)

The high risk precision is 0.01, the sensitivity is 0.70, and the F1 score is 0.02. The the low risk precision is 1.00, the sensitivity is 0.57, and the F1 score is 0.73.

### BalancedRandomForestClassifier
![BRFC](https://github.com/RyleeJensen/Credit_Risk_Analysis/blob/main/Images/BRFC.png)

The high risk precision is 0.04, the sensitivity is 0.67, and the F1 score is 0.07. The low risk precision is 1.00, the sensitivity is 0.91, and the F1 score is 0.95.

### EasyEnsembleClassifier
![EEC](https://github.com/RyleeJensen/Credit_Risk_Analysis/blob/main/Images/EEC.png)

The high risk precision is 0.07, the sensitivity is 0.91, and the F1 score is 0.14. The low risk precision is 1.00, the sensitivity is 0.94, and the F1 score is 0.97.

## Summary
All of the models that we used in this analysis showed weak precision in determining high credit risk. Out of all the models, the EasyEnsembleClassifer model showed the best classification report, especially in terms of sensitivity. The high credit risk was 91%, meaning it can detect almost all high credit risk. But because the precision of the high credit risk is quite low at 0.07, this means there are a lot of false negatives in the model (low credit risks are falsely detected as high credit risks). All these models show the same problems in terms of low precision, so all of them will predict a high number of false negatives. This could negatively impact the bank's credit strategy and possibly even defer future business opportunites if the models all predict high credit risk (even if there are none). For these reasonings, I would not recommend any of the models to be used for assessing credit risk.
