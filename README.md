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
   
<br>ML  Model Name -------------	Accuracy	------AUC	--------Precision --------	Recall	----------F1-------------	MCC
<br>Logistic Regression 	|0.866213	| 	0.807804		|0.700000	|	0.295775		|0.415842		|0.396222	|
<br>Decision Tree        | 0.768707	|0.554853		|0.261538	|	0.239437	|	0.250000	|	0.113740	|
<br>KNN                  | 0.843537 	|	0.662752		|0.571429 	|	0.112676	|	0.188235 	|	0.202208	|
<br>Naive Bayes          | 0.764172	|0.759802	|	0.368000	|	0.647887	|	0.469388	|	0.354238	|
<br>Random Forest        | 0.825397	|	0.746022	|	0.363636	|	0.112676	|	0.172043	|	0.126338	|
<br>XGBoost              | 0.854875	|	0.735287	|	0.629630	|	0.239437	|	0.346939	|	0.325628	|

Observations:

<br>ML Model Name	--------------------------Observation about model performance
<br>Logistic Regression	-----------------Logistic Regression achieved the highest accuracy but very low recall, indicating the model predicts the majority class more often  &ensp and fails to detect many employees who leave. Hence it is biased toward non-attrition cases.	|
<br>Decision Tree	----------------------Decision Tree showed unstable performance with low MCC and F1 score, suggesting overfitting and poor 
 &ensp generalization on unseen data.	|
<br>kNN	--------------------------------KNN produced high accuracy but extremely low recall, meaning it struggles to identify minority attrition cases and is sensitive to  &ensp class                                       imbalance.	|
<br>Naive Bayes----------------------	Naive Bayes achieved the highest recall and F1 score, effectively detecting employees likely to leave. It performed best for the  &ensp imbalanced                                   dataset and provided the most balanced classification.	|
<br>Random Forest (Ensemble)	-------------Random Forest improved stability over Decision Tree but still had low recall, indicating it predicts majority class more  &ensp frequently                                           despite ensemble learning.	|
<br>XGBoost (Ensemble)------------------	XGBoost showed strong accuracy and good overall performance but lower recall than Naive Bayes, meaning it is more conservative in                                              &ensp predicting attrition cases.	|

