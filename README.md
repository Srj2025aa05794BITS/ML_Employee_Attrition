# ML_Employee_Attrition
ML_Employee_Attrition

Problem Statement
Predict whether an employee will leave the organization using classification algorithms.

Dataset
IBM HR Analytics Employee Attrition Dataset
Contains employee demographic, job role and satisfaction features.

Models Implemented
1.	Logistic Regression
2.	Decision Tree
3.	KNN
4.	Naive Bayes
5.	Random Forest
6.	XGBoost
   
ML  Model Name -------------	Accuracy	------AUC	--------Precision --------	Recall	----------F1-------------	MCC
Logistic Regression 	0.866213 	0.807804	0.700000	0.295775	0.415842	0.396222
Decision Tree         0.768707	0.554853	0.261538	0.239437	0.250000	0.113740
KNN                   0.843537	0.662752	0.571429	0.112676	0.188235	0.202208
Naive Bayes           0.764172	0.759802	0.368000	0.647887	0.469388	0.354238
Random Forest         0.825397	0.746022	0.363636	0.112676	0.172043	0.126338
XGBoost               0.854875	0.735287	0.629630	0.239437	0.346939	0.325628

Observations:

ML Model Name	--------------------------Observation about model performance
Logistic Regression	-----------------Logistic Regression achieved the highest accuracy but very low recall, indicating the model predicts the majority class more often and                                         fails to detect many employees who leave. Hence it is biased toward non-attrition cases.
Decision Tree	----------------------Decision Tree showed unstable performance with low MCC and F1 score, suggesting overfitting and poor generalization on unseen data.
kNN	--------------------------------KNN produced high accuracy but extremely low recall, meaning it struggles to identify minority attrition cases and is sensitive to class                                       imbalance.
Naive Bayes----------------------	Naive Bayes achieved the highest recall and F1 score, effectively detecting employees likely to leave. It performed best for the imbalanced                                   dataset and provided the most balanced classification.
Random Forest (Ensemble)	-------------Random Forest improved stability over Decision Tree but still had low recall, indicating it predicts majority class more frequently                                           despite ensemble learning.
XGBoost (Ensemble)------------------	XGBoost showed strong accuracy and good overall performance but lower recall than Naive Bayes, meaning it is more conservative in                                               predicting attrition cases.

