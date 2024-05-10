# Abstract
In 2023, there were 426,000 cases of credit card fraud reported to the FTC which is up 53% from 2019 ^1^. Credit card fraud comes in two main forms: new account fraud where an indentity theif opens a new account in someone else's name and existing account fraud in which an identity theif uses an account that was already open. This project focuses on the latter of the two, working to identify fraudulent transactions on customer's credit cards. With only 0.173% of the dataset as positive cases (fraud), there is a major class imbalance that poses difficulty when training a machine learning model. Synthetic minority oversampling using SMOTE and ADASYN is used to produce fake data to train the initial model on. Several different classification models are fitted and assessed on the model to determine the best results. The final model is verified with a K-Fold Cross Validation.

# Introduction
In 1899 a livestock farmer received a credit card in the mail, but threw it away because he was not interested in using credit. Someone else picked up the credit card and began spending copious amounts of money on luxury transportation. The livestock farmer ended up incurring all of the charges at the end of the month and this case was recorded as the first case of credit card fraud in the United States ^2^. In 2023, there were 426,000 cases of credit card fraud reported to the FTC which is up 53% from 2019 ^1^. Credit card fraud comes in two main forms: new account fraud where an indentity theif opens a new account in someone else's name and existing account fraud in which an identity theif uses an account that was already open. This project focuses on the latter of the two, working to identify fraudulent transactions on customer's credit cards. Through the implementation and assessment of several machine learning models, I aim to be able to detect fraudulent transactions in the dataset, while minimizing false negatives and false positives.

# Description of the Data
The dataset that I am using is from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) ^3^. The data was collected over two days in September 2013, describing 284,807 credit card transactions made by European cardholders.

A Principle Component Analysis (PCA) transformation and dimensionality analysis was used to obtain the features 'V1', 'V2', - ..., - 'V28'. To protect the information of the cardholders and maintain confidentiality, no more information about these features was provided by the dataset creator. 'Time' is the amount of time that has elapsed since the first transaction in the dataset, in seconds. 'Amount is the amount of money charged in each transaction.

![The first 10 rows of the dataset.](/images/dataframe.png.png)

## Class Imbalance
The most challenging part of this dataset was the class imbalance between fraudulent and genuine transactions. There are 284,315 real charges (y = 0) and only 492 fraud charges (y = 1). This means that for every fraudulent charge there are 588 real charges and that only 0.17305% of the dataset is the class that we are trying to detect.

<img src="/images/beforesmote.png.png" alt="Bar graph showing the extreme class balance" width="600"/>

# Methods
## Preprocessing
Checking for null values:

<img src="/images/null_values.png.png" alt="drawing" width="200"/>

### SMOTE and ADASYN to Address the Class Imbalance
I am using SMOTE (Synthetic Minority Oversampling Technique) and ADASYN ( Adaptive Synthetic Sampling) to address the class imbalance that this dataset presents. SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line ^4^. ADASYN focuses on the minority instances that are difficult to classify correctly, rather than oversampling all minority instances uniformly. It assigns a different weight to each minority instance based on its level of difficulty in classification ^4^. While the two minority oversampling techniques have different approaches, due to the extremity of the class imbalance and the fact that they each had to create nearly 600 new points for every point in the original data, they ended up producing nearly identical results. For thoroughness, I will test both SMOTE and ADASYN on each classification model.

Here are some examples of SMOTE creating new points. The original data points are in blue and the newly created points are in red: 

|<img src="/images/smote_pts1.png.png" alt="" width="300"/>|<img src="/images/smote_pts2.png.png" alt="" width="300"/>|
|---|---|
|<img src="/images/smote_pts3.png.png" alt="" width="300"/>|<img src="/images/smote_pts4.png.png" alt="" width="300"/>|

### Scaling
I scaled all of the X features using the standard scaler from Sklearn's preprocessing module. Even with the scaling, there are several outliers, all in the non-fraud class, that are still nearly 30 standard deviations from the mean. I chose not to clip the features as the models were still performing well, but it is something that I kept in mind when training. 

## Visualizing the Data
To see the difference in the data before and after applying SMOTE/ADASYN, I created correlation matrices and used heat maps to visualize them:

|<img src="/images/corr_matrix_og.png.png" alt="" width="500"/>|<img src="/images/corr_matrix_smote.png.png" alt="Heat map of correlation matrix after applying SMOTE" width="500"/>|
|---|---|

---

Scatterplots showing fraud transactions (red) and real transactions (blue) for different variables:

|<img src="/images/V1V2.png.png" alt="V1 vs V3" width="700"/>|<img src="/images/V1V3.png.png" alt="V1 vs V2" width="700"/>|
|--|--|

## Classification Methods
I decided to try four different classification methods to try to get the best results for this data. For this problem, it is important to detect all positive cases (y = 1), as these are fraud cases. Obviously we want to minimize both false positives and false negatives, however missing a fraudulent transaction, given how rare they are, is a worse error than accidentally flagging a real charge as fraud. Because less than a fifth of a percent of the original data set are positive cases, if the model misclassified all of them as real charges, the accuracy would still be over 99.8%. To combat this, I will be using recall score, precision score, and confusion matrix in addition to accuracy to check the performance of each model. 

**Recall = true positives / (true positives + true negatives)**

<img src="/images/recall_diagram.png" alt="" width="700"/>
source: https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9

**Precision = true positives / (true positives + false positives)**

<img src="/images/precision_diagram.png" alt="" width="700"/> 
source: https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9

--- 

## K-Nearest Neighbor

<img src="/images/knn_diagram.png" alt="" width="500"/>
source: https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning

|   |SMOTE|ADASYN|
|---|---|---|
|accuracy|0.9621|0.96134| 
|recall|0.97857|0.97971
|precision|0.94744|0.94516

**Confusion Matrices:**

|<img src="/images/cf_knn_smote.png" alt="SMOTE" width="400"/>|<img src="/images/cf_knn_adasyn.png" alt="SMOTE" width="400"/>|
|--|--|

---

## Decision Tree

<img src="/images/tree_diagram2.png" alt="" width="400"/>
source: https://www.smartdraw.com/decision-tree/


max_depth = 5

|   |SMOTE|ADASYN|
|---|---|---|
|accuracy|0.96844|0.96848| 
|recall|0.95801|0.95637 
|precision|0.98122|0.98016

**Confusion Matrices:**

|<img src="/images/cf_tree_smote.png" alt="SMOTE" width="400"/>|<img src="/images/cf_tree_adasyn.png" alt="ADASYN" width="400"/>|
|--|--|

---

## Logistic Regression

<img src="/images/lr_diagram.png" alt="" width="400"/>
source: https://medium.com/analytics-vidhya/the-math-behind-logistic-regression-c2f04ca27bca

|   |SMOTE|ADASYN|
|---|---|---|
|accuracy|0.95797|0.97039| 
|recall|0.94805|0.96181
|precision|0.96741|0.97868

**Confusion Matrices:**

|<img src="/images/cf_lr_smote.png" alt="SMOTE" width="400"/>|<img src="/images/cf_lr_adasyn.png" alt="ADASYN" width="400"/>|
|--|--|

---

## Random Forest

<img src="/images/rf_diagram.png" alt="" width="400"/>
source: https://www.researchgate.net/figure/Schematic-diagram-of-the-random-forest-algorithm_fig3_355828449

n_estimators = 100

|   |SMOTE|ADASYN|
|---|---|---|
|accuracy|0.99991|0.99990| 
|recall|1.00000|1.00000
|precision|0.99982|0.99981

**Confusion Matrices:**

|<img src="/images/cf_rf_smote.png" alt="SMOTE" width="400"/>|<img src="/images/cf_rf_adasyn.png" alt="ADASYN" width="400"/>|
|--|--|

---

## Validation
Based off of the results above, the best model for this data is the random forest with SMOTE. K-Fold Cross Validation is a great method for ensuring that a model is not overfit and to test it on more than one test set with the same amount of data. I perform a K-Fold Cross Validation with k = 6: 

<img src="/images/kfcv.png" alt="K-Fold Cross Validation Diagram with k=6" width="500"/>

|Split  | Precision |
|--|--|
|1|0.98818|
|2|0.99983|
|3|0.99987|
|4|0.99932|
|5|0.99966|
|6|0.99971|

# Discussion and Inferences
Based on the results of this project, I would use the Random Forest classifier trained on the SMOTE data to predict if future transactions are fraudulent or genuine. The model caught all of the fraud cases and misclassified only .000017% of the real transactions as fraud. It is likely that after transactions are classified as fraud, they are reviewed by a human or possibly the card holder before any action is taken, so while ideally there would be 0 false positives, it is still a very low number and shows that the model is performing well. I would like to test this model on more data to see how it performs, especially because this data is such a small subset of all credit card transactions and is from a small time frame (two days) and only from European cardholders.

## References
1. Caporal, J. (2024, February 29). Identity Theft and Credit Card Fraud Statistics for 2024. The Motley Fool. Retrieved May 10, 2024, from https://www.fool.com/the-ascent/research/identity-theft-credit-card-fraud-statistics/
2. McKenna, F. (2022, July 22). The Story of the Very First Case of Credit Card Fraud. Frank on Fraud. Retrieved May 10, 2024, from https://frankonfraud.com/fraud-trends/first-credit-card-fraud-case-was-in-1899/
3. Machine Learning Group - ULB. (n.d.). Credit Card Fraud Detection. Kaggle. Retrieved May 10, 2024, from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
4. M, V. (2023, September 25). Data Imbalance: How is ADASYN different from SMOTE? Medium. Retrieved May 10, 2024, from https://medium.com/@penpencil.blr/data-imbalance-how-is-adasyn-different-from-smote-f4eba54867ab

