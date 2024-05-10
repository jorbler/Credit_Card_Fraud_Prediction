# Abstract

  

# Introduction

In 1899 a livestock farmer received a credit card in the mail, but threw it away because he was not interested in using credit. Someone else picked up the credit card and began spending copious amounts of money on luxury transportation. The livestock farmer ended up incurring all of the charges at the end of the month and this case was recorded as the first case of credit card fraud in the United States.

  

# Description of the Data

The dataset that I am using is from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The data was collected over two days in September 2013, describing 284,807 credit card transactions made by European cardholders.

  

A Principle Component Analysis (PCA) transformation and dimensionality analysis was used to obtain the features 'V1', 'V2', - ..., - 'V28'. To protect the information of the cardholders and maintain confidentiality, no more information about these features was provided by the dataset creator. 'Time' is the amount of time that has elapsed since the first transaction in the dataset, in seconds. 'Amount is the amount of money charged in each transaction.

  

![The first 10 rows of the dataset.](/images/dataframe.png.png)

  

## Class Imbalance

The most difficult part of this dataset was the class imbalance between fraudulent and genuine transactions. There are 284,315 real charges (y = 0) and only 492 fraud charges (y = 1). This means that for every fraudulent charge there are 588 real charges and that only 0.17305% of the dataset is the class that we are trying to detect.

  

![Bar graph showing the extreme class balance.](/images/beforesmote.png.png)

  

## Exploratory Data Analysis

  
  

# Methods

## Preprocessing

Checking for null values:

![](/images/null_values.png.png)

  

### SMOTE and ADASYN for Minority Oversampling

I am using SMOTE (Synthetic Minority Oversampling Technique) and ADASYN ( Adaptive Synthetic Sampling) to address the class imbalance that this dataset presents. SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line [^1]. ADASYN focuses on the minority instances that are difficult to classify correctly, rather than oversampling all minority instances uniformly. It assigns a different weight to each minority instance based on its level of difficulty in classification [^2]. While the two minority oversampling techniques have different approaches, due to the extremity of the class imbalance and the fact that they each had to create nearly 600 new points for every point in the original data, they ended up producing nearly identical results. For thoroughness, I will test both SMOTE and ADASYN on each classification model.

  

Here are some examples of SMOTE creating new points. The original data points are in blue and the newly created points are in red:

  
![](/images/smote_pts1.png.png)

![](/images/smote_pts2.png.png)

![](/images/smote_pts3.png.png)

![](/images/smote_pts4.png.png)

  

### Scaling

I scaled all of the X features using the standard scaler from Sklearn's preprocessing module.

  

## Visualizing the Data

  

To see the difference in the data before and after applying SMOTE/ADASYN, I created correlation matrices and used heat maps to visualize them:

  

![](/images/corr_matrix_og.png.png)

  

![Heat map of correlation matrix after applying SMOTE](/images/corr_matrix_smote.png.png)

  

Scatterplots showing fraud transactions (red) and real transactions (blue) for different variables:

  

![V1 vs V3](/images/V1V2.png.png)

  

![V1 vs V2](/images/V1V3.png.png)
