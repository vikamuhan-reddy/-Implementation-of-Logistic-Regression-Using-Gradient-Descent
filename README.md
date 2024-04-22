# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Initialize Parameters
2. Compute Predictions
3. Compute Loss
4. Compute Gradients
5. Update Parameters
6. Continue the Process
7. Evaluate and Predict the data

## Program:
```py
'''
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vikamuhan reddy.N
RegisterNumber:  212223040181
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("D:/introduction to ML/jupyter notebooks/Placement_Data.csv")
print(dataset)

#dropping the serial no and salary col

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
print(dataset)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
print(dataset.dtypes)

dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes

print(dataset)

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
#Initialize the model parameters
theta = np.random.randn(X.shape[1])
y = Y
#Define the sigmoid function:
def sigmoid(z):
    return 1 / (1+np.exp(-z))
# define the loss function:
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

#define the gradient descent algorithms
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y)/m
        theta -= alpha*gradient
    return theta

#train the model:
theta = gradient_descent(theta,X,y,alpha = 0.01,num_iterations = 1000)

# make predictions:
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5,1,0)
    return y_pred

y_pred = predict(theta,X)
print(y_pred)

Accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy: ",Accuracy)
```
## output:

# dataset:
![image](https://github.com/vikamuhan-reddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144928933/4d81269f-3b76-4c9f-b739-fedfa2bce5b8)

# dataset.dtypes:
![image](https://github.com/vikamuhan-reddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144928933/aba75df7-8384-46a4-8569-b57784f00c4f)

# dataset after catcodes:
![image](https://github.com/vikamuhan-reddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144928933/34b8ca4f-99e2-495d-9e02-534b26fb18f3)

# Accuracy:
![image](https://github.com/vikamuhan-reddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144928933/f83af9e4-a26c-4c0e-96ac-40e67fedb68e)

# y_predicted value:
![image](https://github.com/vikamuhan-reddy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144928933/04a9f737-8c06-4b75-87ce-322e17f01598)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

