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
