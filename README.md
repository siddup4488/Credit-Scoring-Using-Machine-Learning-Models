# Credit-Scoring-Using-Machine-Learning-Models

Description:

Implement supervised machine learning techniques in order to further understanding the process in which a client will be granted a credit and be denied a credit. This process is denoted as credit scoring, it is a wide methodology used by banks which assigns each prospect client a score from 300 to 850, being 850 the highest score a client can receive. Credit scoring is used to evaluate the potential risk that granting a client a credit poses on credit lenders. A credit score is based on an individual' credit report, which considers both numerical and categorical variables, such as the status of the existing credit account, the credit amount, number of existing credits at the bank, among others. Ultimately, credit lenders use such score to determine which clients will be granted credit loan under a predetermined interest rate and credit limit.

Introduction

As computers are required to solve problems of higher complexity, tasks arise where traditional programming approaches cannot be used. Such scenarios happen when the designer
of the system cannot correctly declare a method that uses input data to compute a correct output. Therefore, as stated in (1), an approach in which a computer “attempts to
learn the input/output functionality” is implemented to further understand and classify the
data. As stated in (5), during the past decades, credit scoring has proved to be ”one of the
most successful applications of statistical and operations research modeling in finance and
banking.” Credit scoring practices have allowed financial institutions in the credit sector
to witness an increase in their capital and prospect money investment when dealing with
costumer credit. It is detrimental for financial institutions to have the ability to identify
what types of credit requests fall under the category of clients that are likely to repay
their debt and those who aren’t. To do so, ”the method produces a “score” that a bank
can use to rank its loan applicants or borrowers in terms of risk (9).” As denoted in (9),
the score produced by the method takes into account information such as ”the applicant’s
monthly income, outstanding debt, financial assets, how long the applicant has been in
the same job, whether the applicant has defaulted or was ever delinquent on a previous
loan, among others.” The higher the score, the lower the risk associated with the applicant
and vice versa. Therefore, credit scoring serves as a tool to mitigate the risk that financial
institution face when granting a credit to their clients .

2 Methods

2.1 Logistic Regression

The logistic regression method is a supervised Machine Learning algorithm, which can be
used for regression problems or as a classification method and because it does not require
2
too many computational resources, it turns out to be one of the most implemented algorithms nowadays in fields like economy, finance, epidemiology, among others. This allows
us to estimate the relationship between a group of explanatory variables (Xi, i=1, ..., n) and
a binary response variable Y in terms of the probability of occurrence of the two possible
classification groups.
Actually, this method consists of two parts. Initially it can be understood as a linear
method if we consider the logit of Y, that is the logarithm of the probabilities that Y is
equal to one of the two categories

![image](https://user-images.githubusercontent.com/61466844/230624546-6485b799-8a3a-4cf6-8d58-005faffcf38f.png)


For which the regression coecients  must be estimated from the maximum likelihood
method and with the p-values granted by the test, in an iterative way, those variables that
present the highest of these will be eliminated until only those that have a p-value lower
than 0.05 are obtained, since in this way the correlation between the explanatory variables
would be eliminated and a better adjustment of the model would be achieved. Finally,
we proceed to eliminate the linearity from the inverse of the logistic function and thus be
able to obtain a behavior in the form of S, where the maximum value will be one and the
minimum will be zero, because we are calculating the probability of permanence to one of
the two groups, in which one can be considered as success and the other as failure.

![image](https://user-images.githubusercontent.com/61466844/230624621-b5852a71-1286-4283-acdc-7ad2713254f8.png)

Therefore, it turns out to be a linear model for which the  parameters have to be estimated by implementing the maximum likelihood method. Now, the ratio of the probability
of success to the probability of failure, known as odds, proves to be very useful, as it allows
us to understand the relationship between individuals who depend or do not depend on a
risk factor. In addition, the odds ratio provides the odds ratio (OR), which indicates the
proportion in which the odds of being subject to a factor exceeds the odds of not being
subject to it. Then, in order to classify the data in the two groups given by the binary or
dichotomous variable, a cut-o↵ point is established for the values obtained with the logit
function and those that exceed this value will belong to one group and the others to the
other, in most of the cases the cuto↵ point is 0.5.
