+++
title = 'Logistic Regression'
date = 2022-12-22T13:08:21+05:30
draft = false
author = "Ranjit M"
tags =["Logistic Regression", "Machine Learning", "Binary Classification", "Data Science", "Statistical Modeling","Python","Gradient Descent","Sigmoid Function","Cost Function"]
+++
# Logistic Regression

Logistic Regression is a statistical method used for binary classification. It predicts the probability of a binary response based on one or more predictor variables.

![Logistic Regression Image](/images/logistic-reg.jpeg)

## Key Concepts

### 1. Sigmoid Function
The sigmoid function maps predicted values to probabilities:
\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]
where \( z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n \).

The sigmoid function outputs a value between 0 and 1, which can be interpreted as a probability.

### 2. Hypothesis Function
In logistic regression, the hypothesis function \( h_\theta(x) \) is defined as:
\[ h_\theta(x) = \sigma(\theta^T x) \]
This function outputs the estimated probability that \( y = 1 \) given input \( x \).

### 3. Cost Function
The cost function used in logistic regression is the log-loss or binary cross-entropy loss:
\[ J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] \]
This function measures the performance of the model by comparing the predicted probabilities with the actual class labels.

### 4. Gradient Descent
Gradient Descent is used to minimize the cost function:
\[ \theta := \theta - \alpha \frac{\partial J(\theta)}{\partial \theta} \]
where \( \alpha \) is the learning rate. The gradient of the cost function with respect to the parameters \( \theta \) is computed and used to update the parameters iteratively.

## Mathematical Formulation

1. **Logistic Model**:
   \[ P(y=1|x; \theta) = \sigma(\theta^T x) \]
   \[ P(y=0|x; \theta) = 1 - \sigma(\theta^T x) \]

2. **Decision Boundary**:
   The decision boundary is defined by the set of points where \( \theta^T x = 0 \).

3. **Parameter Estimation**:
   Parameters \( \theta \) are estimated using Maximum Likelihood Estimation (MLE). The likelihood function is:
   \[ L(\theta) = \prod_{i=1}^{m} P(y^{(i)}|x^{(i)}; \theta) \]
   Taking the logarithm of the likelihood function gives the log-likelihood:
   \[ \ell(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] \]

## Python Implementation

Here's a basic implementation of logistic regression using Python and the popular library `scikit-learn`.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv('data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
