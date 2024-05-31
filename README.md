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
