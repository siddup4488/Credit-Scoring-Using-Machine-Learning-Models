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

Some of the advantages of this algorithm are:
1. It is an ecient and easy to implement algorithm because it does not require too
many computer resources.
2. It eliminates assumptions made in the linear regression method such as the normality
of the data under consideration.
3. It allows better adjustment of the data to be categorized according to a binary response variable.
However, it also has its disadvantages such as:
1. It cannot be applied to non-linear problems because it is based on linear regression.
2. It requires a good pre processing of the data in which those variables that are highly
correlated with each other must be eliminated.

 Pseudocode
Algorithm 1 Logistic Regression
data Load data
X explanatory variables
Y response variable
ypred initialize vector
pv true
while pv is true
for all xi vector in X do
i) Calculate the regression coecients using the maximum likelihood method.
end
ii) Find the maximum p-value
if maximum p-value > 0.05 then
iii) Delete variable xi from X
else pv = false
end
end
iv) Apply the inverse logit function that will give you a probability p for each observation.
for all p do
if p >= 0.5 then
ypred = 1
else ypred = 0 end end


2.2 K-Nearest Neighbor
KNearest Neighbor is a supervised Machine Learning algorithm which can be used for
classification problems and as regression predictive problem. This method is very simple,
nevertheless, its results may be highly accurate. It is a non-parametric method, has a lazy
learning and it0
s mostly used for recognition of patterns, data mining, anomaly detection,
economics, bank systems and calculating credit ratings.
In Table 1, we compare 3 important aspects scoring them from 1 to 3, being 3 the best
answer to each aspect. We can observe that KNN is mainly used for its easy interpretation
and low calculation time. Srivastava 2018

![image](https://user-images.githubusercontent.com/61466844/230625143-6c37f132-93e3-413c-9b70-e2fdc4a93bd0.png)

Given a dataset, the K-NN algorithm consists in selecting a k value and in the moment
of analysis the k nearest points to the desire class will be the solution. The most important
phase of the method is the k value selection, it must be selected accordingly with the
dataset we are working with.
For better understanding here is an example: lets say we have two classes, C1red
circles and C2green squares. Now we want to classify the Blblue star as seen in Figure

![image](https://user-images.githubusercontent.com/61466844/230625219-be70389f-186e-49bd-beec-548d4f1082b5.png)


Bl can be either from C1 or C2. Let’s say that k = 3, now we take Bl as the centroid
of a circle big enough that encloses k data points as seen in Figure 2.

![image](https://user-images.githubusercontent.com/61466844/230625283-a21f14fe-c15c-4401-8b07-057667209222.png)

The 3 closest points to Bl are C1, then we can say that it belongs to that class. Nevertheless, when the circle encloses more than one class points, it would classify Bl from the
class that has more points in the circle. This is the reason why k is preferred to be a odd
number. Nevertheless, as every algorithm, it has its advantages and disadvantages.

Advantages:
1. The algorithm is simple and easy to implement.
2. There’s no need to build a model, tune several parameters, or make additional assumptions.
3. The algorithm is versatile. It can be used for classification, regression, and search.

Disadvantages:
1. The algorithm gets significantly slower as the number of examples and/or predictors/independent variables increase

Pseudocode
Algorithm 2 K-nearest neighbor
data Load data
k initialize value
for all data point do
i) Calculate distance between the query and the current example from the data.
ii) Add the distance and the index of the example to an ordered collection.
iii) Sort the ordered collection of distances and indices in ascending order.
iv) Pick the first K entries from the sorted collection.
v) Get the labels of the selected K entries.
if regression then
return the mean of the K labels
else
return the mode of the K labels.
end

2.3 Support Vector Machine
 Support Vector Machines is a discriminative classifier whose classification is based on the construction, in a high or infinite
dimensional space, of a hyperplane or a set of hyperplanes. By definition, the hyperplane
is used as a decision boundary in which each input vector from the input space is classified.
This algorithm can be used for a variety of purposes: classification, regression and outlier
detection.
2.3.1 Linear Classification
a binary classification problem involves
the use of a classification decision rule to classify elements into two distinct groups. For
notation purposes let X 2 Rn, denote the input space and Y = {1, 1} denote the output
domain. Let an example from the data set be represented as the pair (xi, yi), where
xi = (x1, .., xn) represents the input vector and yi the respective output value. Lastly, let
the training data be denoted as S = ((x1, y1)), ...,(xl, yl)) 2 (X, Y )
l and let the testing data
be denoted as S0 = ((x1, y1), ...,(xk, yk)) 2 (X, Y )
k
.
Given a testing example (xi, yi) 2 S0
, binary classification is performed by using a
function f : X 2 Rn ! R, denoted as

![image](https://user-images.githubusercontent.com/61466844/230625937-c3b6cab3-3b7a-4aec-a65f-4c731a50f5a6.png)

2.3.2 Kernel-Induced Feature Spaces
In some cases, the linear combination of independent variables cannot e↵ectively classify the
response variable. In this case, the idea of more abstract features of the data is considered.
According to Cristianini and Shawe-Taylor (2000), by using the kernel representation of the
data, linear classification learning machines can better classify data on a high dimensional
feature space.
The purpose of kernel representation is to change the way data is represented in order to
obtain an objective function that can fully distinguish di↵erent classes. The preprocessing
method consist on changing the representation of all xi 2 X i = {1, ..., n}:
xi = (x1, ..., xn) ! (xi)=(1(xi), ..., d(xi)) d  n. (3)
The process represented in (3) is equivalent to mapping the input space X into a
feature space represented by F = {(xi)|xi 2 X} . Let  : X ! F represent a non-linear
map from the input space to a feature space.
A reason why feature mapping occurs is due to the need of machines to classify nonlinear relationships. Feature mapping allows non-linearly separable data to become linearly
separable by rewriting the data into a new representation. According to Cristianini and
Shawe-Taylor (2000), this process consists on using a linear machine to classify the data
on the feature space, which was initially applied a non-linear mapping. Given a testing
example (xi, yi) 2 S0
, the linear machine used to classify the testing example on feature
space F is denoted as
f(xi) = hwT · (xi)i + b, (4)
where (w, b) 2 Rd ⇤ R are parameters that control the decision rule in the feature space
and are learned from a S.
Due to the fact that linear learning machines can also be expressed in dual representation, (4) can be rewritten as the inner product between the input vector from a testing
example (xi, yi) 2 S0 and the input vector from the training points in S in the feature space
F:
f(xi) = X
l
j=1
↵jyj h(xj ) · (xi)i + b. (5)
According to Cristianini and Shawe-Taylor (2000), for all pairs of examples in the input
space, a kernel function, K, can be represented as:
K(x, z) = h(x) · (z)i.

