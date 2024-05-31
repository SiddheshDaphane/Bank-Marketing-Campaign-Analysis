# Bank-Marketing-Campaign-Analysis, Unleashing Insights with Machine Learning

## Project Motivation:
Banks need to attract new customers and retain their existing customers to maintain profitability. However, traditional marketing campaigns can be expensive and often fail to deliver desired results. To address this challenge, banks are seeking efficient ways to target their customers and improve their marketing campaigns. One such approach is the use of machine learning techniques.
This project aims to explore how machine learning can be leveraged to analyze a bank marketing dataset and identify factors that contribute to a successful campaign. Our primary goal is to develop predictive models that can accurately predict customer subscription to the bank's term deposit. In doing so, we aim to compare the performance of different machine learning algorithms and develop a dashboard for visualizing the results.
By achieving our objectives, we hope to provide valuable insights to banks on optimizing their marketing campaigns. Additionally, we aim to demonstrate the power of machine learning in solving real-world business problems and inspire future research in this field.

## Dataset Description:
The project deals with a second-hand data set (Bank Marketing Data Set) [1][2]. It is extracted from the Machine Learning Repository of the University of California, Irvine. The data features real-time statistics collected over phone calls from 2008 to 2013. Collected in a marketing ad campaign, a Portuguese banking establishment gathers the following information. The following data encompasses the socio-economic details of clients who were contacted under a telemarketing campaign for selling the bank's long-term deposits. The dataset contains information about a bank's marketing campaigns, including customer demographic data, contact information, and campaign outcomes. The dataset has 21 features and 41211 rows which include information as below:

• Age (numeric): the age of the client

• Job (type of job): the type of job the client has

• Marital (married/divorced/single): the marital status of the client

• Education (unknown, secondary, primary, tertiary): the education level of the client

• Housing: whether the client has a housing loan or not

• Loan: whether the client has a personal loan or not

• Duration: the duration of the last contact, in seconds

• y - has the client subscribed to a term deposit? (Binary: 'yes', 'no')

Potential targets: The target variable is whether the client subscribed to the term deposit. This variable is represented by the "y" column in the dataset, which takes on a value of "yes" or "no"

## Methodology:
The project will involve several steps to accomplish its objectives. Firstly, the data will be loaded, and an Exploratory Data Analysis (EDA) will be conducted to gain insights into the dataset. Based on the findings of the EDA, hypotheses will be formulated regarding the individual factors (features) in the dataset. This will guide the data cleaning and preparation process for modeling.
Next, the metrics to be used for evaluating the model will be chosen. A pipeline for Cross Validation and Grid Search procedures will then be built to search for the optimal parameters of the model. The most effective model will be chosen based on the evaluation results, and a learning curve rate will be built to assess the model's performance.
Finally, conclusions will be drawn based on the results, and the methodology will be modified and improved as necessary.

## Interesting key takeaway:
Based on the performances of the different models on the different datasets, it can be concluded that the Random Forest Classifier and XGBoost models are the best-performing models among the models tested. Both models have high accuracy and F1-score, and consistent cross-validation performance. However, the precision and recall for the minority class (adopters for the Random Forest Classifier and class 1 for XGBoost) are relatively low, indicating that the models may not perform as well in identifying the minority class.
The Decision Tree model and KNN model also show promise, but there is room for improvement in identifying the minority class accurately. The Logistic Regression model and Naive Bayes classifier perform relatively well, but they may not be the best options for the datasets used since they have lower accuracy and F1-scores than the other models

## Introduction:
Marketing is a critical aspect of business success, and banks are no exception. To maintain profitability, banks need to attract new customers while retaining their existing customers. One of the common ways to achieve this is through effective marketing campaigns. Traditional marketing campaigns can be costly and often do not produce the desired results. Therefore, banks are looking for more efficient ways to target their customers and make their campaigns more effective. Machine learning provides a powerful tool to achieve this goal.
The primary motivation behind this project is to explore how machine learning techniques can be used to analyze a bank marketing dataset and identify factors that contribute to a successful marketing campaign. The objective is to develop predictive models that can predict whether a customer will subscribe to the bank's term deposit or not. We aim to compare the performance of different machine learning algorithms in predicting customer subscription and develop a dashboard that can be used to visualize the results of the analysis.
The context of this project is to address the challenges faced by banks in optimizing their marketing campaigns. By leveraging machine learning techniques, banks can gain valuable insights into customer behavior and tailor their marketing efforts accordingly. This can result in increased customer acquisition and retention, as well as higher profitability for the bank.
The goals of this project are twofold. First, we aim to identify the key factors that contribute to a successful marketing campaign. Second, we aim to demonstrate the power of machine learning in solving real-world business problems, such as optimizing marketing campaigns. By achieving these goals, we hope to provide valuable insights to banks and inspire future research in this field.

## Datasets and Preprocessing
Source of the data: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

The project deals with a second-hand data set (Bank Marketing Data Set) [1][2]. It is extracted from the Machine Learning Repository of the University of California, Irvine. The data features real-time statistics collected over phone calls from 2008 to 2013. Collected in a marketing ad campaign, a Portuguese banking establishment gathers the following information. The following data encompasses the socio-economic details of clients who were contacted under a telemarketing campaign for selling the bank's long-term deposits. The dataset contains information about a bank's marketing campaigns, including customer demographic data, contact information, and campaign outcomes. The
dataset has 21 features and 41211 rows which include information as below:

• • • • • • • •

Age (numeric): the age of the client

Job (type of job): the type of job the client has

Marital (married/divorced/single): the marital status of the client

Education (unknown, secondary, primary, tertiary): the education level of the client

Housing: whether the client has a housing loan or not

Loan: whether the client has a personal loan or not

Duration: the duration of the last contact, in seconds

y - has the client subscribed to a term deposit? (Binary: 'yes', 'no')Potential targets

The target variable is whether the client subscribed to the term deposit. This variable is represented by the "y" column
in the dataset, which takes on a value of "yes" or "no".


## EXPLORATORY DATA ANALYSIS
To understand the data, we performed below statistical analysis:

1. Checking the data distribution of each Continuous variable:


By creating a histogram for each variable in the dataset, we can quickly identify any patterns or anomalies in the data.

<img width="1037" alt="Screenshot 2024-05-31 at 8 20 03 AM" src="https://github.com/SiddheshDaphane/Bank-Marketing-Campaign-Analysis/assets/105710898/1bd1dd7c-881b-4692-80b7-40a3c1a5dff8">


2. Boxplot

<img width="957" alt="Screenshot 2024-05-31 at 8 20 57 AM" src="https://github.com/SiddheshDaphane/Bank-Marketing-Campaign-Analysis/assets/105710898/19811a29-58e1-4ff7-ab69-6afed06f179d">

3. Heatmap

<img width="1007" alt="Screenshot 2024-05-31 at 8 21 56 AM" src="https://github.com/SiddheshDaphane/Bank-Marketing-Campaign-Analysis/assets/105710898/9170aeed-2d2f-4188-b617-6ecb2ba33916">


To understand distribution of data performed below actions

1. Distribution of the different education levels of the clients in the dataset:

  <img width="903" alt="Screenshot 2024-05-31 at 8 23 03 AM" src="https://github.com/SiddheshDaphane/Bank-Marketing-Campaign-Analysis/assets/105710898/f79d373d-1962-4d8a-9a38-b26ee3da680d">


2. Count plot of the contact column:

<img width="956" alt="Screenshot 2024-05-31 at 8 24 12 AM" src="https://github.com/SiddheshDaphane/Bank-Marketing-Campaign-Analysis/assets/105710898/8ff5155d-c633-4e68-9fc7-937626183545">

3. Count plot of the loan column

<img width="967" alt="Screenshot 2024-05-31 at 8 24 48 AM" src="https://github.com/SiddheshDaphane/Bank-Marketing-Campaign-Analysis/assets/105710898/1eeb5700-14b2-4038-b1cb-5d7c05b6e2ba">

4. Clients with or without credit in default

<img width="949" alt="Screenshot 2024-05-31 at 8 25 26 AM" src="https://github.com/SiddheshDaphane/Bank-Marketing-Campaign-Analysis/assets/105710898/0f5f7d3b-dc31-47de-8a03-2ccd4d8afc1b">

5. Clients with housing loan/or not

   <img width="809" alt="Screenshot 2024-05-31 at 8 26 08 AM" src="https://github.com/SiddheshDaphane/Bank-Marketing-Campaign-Analysis/assets/105710898/77c265cc-089d-4584-94e8-4a9448d11990">

6. Data Distribution of Target variable

<img width="1027" alt="Screenshot 2024-05-31 at 8 26 43 AM" src="https://github.com/SiddheshDaphane/Bank-Marketing-Campaign-Analysis/assets/105710898/bf416fc4-81b2-4d63-b3f6-a00db22dd56b">

7. Class Distribution of Term Deposit Subscription: Addressing Imbalanced Data with SMOTE

<img width="830" alt="Screenshot 2024-05-31 at 8 27 17 AM" src="https://github.com/SiddheshDaphane/Bank-Marketing-Campaign-Analysis/assets/105710898/2c1485f1-ca2b-4cf6-ac2c-88b90024f165">

8. Bar plot which shows client subscribed a term deposit or not?

<img width="967" alt="Screenshot 2024-05-31 at 8 27 50 AM" src="https://github.com/SiddheshDaphane/Bank-Marketing-Campaign-Analysis/assets/105710898/85940e64-70ed-45c4-a050-6c40d39b749f">

## Data Preparation for ML

Preparing Bank Marketing Campaign Data for Machine Learning: Creating Dummy Variables and Predictor/Response Variables.
We then prepared the data for model training by converting categorical variables to binary features and creating predictor and response variables.
The code uses the pd.get_dummies() function to create dummy variables for categorical features in the datagram. The resulting Data Frame, df1, has a binary column for each unique category in each categorical feature. The drop_first=True argument specifies that one of the binary columns for each feature is dropped to avoid the issue of collinearity between the binary features.
The resulting data frame, df1, is then used to create the predictor variable (X) and the response variable (y) for model training. The predictor variable X contains all the columns of df1 except the response variable (y). The response variable y contains the binary outcome of the campaign, with a value of 1 indicating that the client subscribed to the term deposit and 0 indicating that they did not.

**Feature Selection:**

To select the most important features for our machine learning model, we used the ExtraTreesClassifier algorithm, an ensemble method that fits multiple randomized decision trees and combines their predictions to improve accuracy and reduce overfitting. In this code, the ExtraTreesClassifier is fit on the predictor variable X and the response variable y using 100 estimators.
Next, the SelectFromModel function from scikit-learn is used to select the most important features from the fitted ExtraTreesClassifier. The SelectFromModel function selects the features that have a greater importance score than the mean importance score. The selected features are stored in X_new.
Then split the dataset into training and testing sets using the train_test_split function from scikit-learn. The function splits the X_new and y variables into training and testing subsets with a 75:25 split ratio. The stratify parameter ensures that the class distribution is preserved in the split, and the random_state parameter ensures reproducibility of the results.
Overall, we prepared the dataset for model training and evaluation by splitting it into training and testing subsets.
Here, we realized the features that have more impact on the predictive outcomes of the models are
Age, Duration of last contact, campaign, emp.var.rate, cons.conf.idx, euribor3m, nr.employed, housing_yes, loan_yes, poutcome_success

**SMOTE(Synthetic Minority Oversampling Technique)**

As we know that target variable was imbalanced, with only 11.3% of instances belonging to the positive class. This made us use the SMOTE technique to balance the data.
The SMOTE algorithm (Synthetic Minority Oversampling Technique) is a data sampling technique that is used to address the issue of class imbalance. SMOTE works by generating synthetic instances of the minority class by identifying nearest neighbors and randomly sampling in the feature space.
In this study, the SMOTE algorithm was implemented using the imbalanced-learn package in Python. The SMOTE algorithm was initialized with a sampling strategy of 'auto' and a random state of 42. The fit_resample() method of the SMOTE object was then called to apply the SMOTE algorithm to the training set, which consisted of the input features X_train and the target variable y_train.
The results of applying the SMOTE algorithm to the training set are shown in the following figure. The figure shows the distribution of the target variable in the original training set and the oversampled training set. As can be seen from the figure, the distribution of the target variable in the oversampled training set is much more balanced than the distribution of the target variable in the original training set. This suggests that the SMOTE algorithm was successful in mitigating the issue of class imbalance in the training set.
The results of this study suggest that the SMOTE algorithm is an effective way to address the issue of class imbalance in binary classification datasets. The SMOTE algorithm can be used to generate synthetic instances of the minority class, which can help to improve the performance of machine learning models on imbalanced datasets.


**Feature Scaling**

Here we have used StandardScaler class from the scikit-learn library for feature scaling. It was an important preprocessing step in machine learning that ensures all input features are on the same scale, which can improve the performance of some machine learning algorithms.
In this code, the StandardScaler class is imported from the scikit-learn preprocessing module. Then, two StandardScaler objects are instantiated for the training and test data, respectively. The fit_transform() method is called on the training data StandardScaler object to fit the scaler to the data and then transform the data. The transform() method is called on the test data StandardScaler object to transform the test data using the mean and standard deviation learned from the training data.
The resulting scaled data can be used for machine learning modeling.
