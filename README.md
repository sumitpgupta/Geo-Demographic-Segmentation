# Geo-Demographic-Segmentation
This is the code for a double-layer feedforward neural network that predicts which customers are at the highest risk of leaving a (simulated) bank company.

# Overview
The model is based on a double-layered feed forward Neural Network that predicts which customers are at highest risk of leaving the bank based on simulated bank data.

I built the model using the Keras library, which is built on top of Tensorflow and Theano. The inputs are a mixture of numeric, binary, as well as categorical values, and the output is binary (indicating whether the customer leaves the bank). I used adam for stochastic optimization, and binary crossentropy as the loss function.

# Dependencies
● tensorflow
● keras
● numpy
● pandas
● scikit-learn

# Dataset
The dataset used is a simulated bank data (Churn_Modelling.csv) which has 10,000 customer observations and 13 attributes.

| Variable | Definition |
| -------- | ------------ |
| CustomerId | Customer's account ID |
| Surname |	Customer's surname |
| CreditScore |	Customer's credit score |
| Geography	| Country (France/Germany/Spain) |
| Gender | Customer's gender (Male/Female) |
| Age	| Customer's age |
| Tenure |	Number of years customer has been with the bank |
| Balance |	Customer's account balance |
| NumOfProducts |	Number of bank products used by customer |
| HasCrCard |	Does customer have a credit card? |
| IsActiveMember |	Is the customer an active member? |
| EstimatedSalary	| Customer's estimated salary |
| Exited |	Did the customer leave the bank? |

# Usage
Run `Churn.py` in terminal to see the network in training. I have used Spyder from Anaconda to script and visualize the code.

# Test Run

```
Epoch 94/100
8000/8000 [==============================] - 0s - loss: 0.3996 - acc: 0.8356      
Epoch 95/100
8000/8000 [==============================] - 0s - loss: 0.3997 - acc: 0.8352     
Epoch 96/100
8000/8000 [==============================] - 0s - loss: 0.4000 - acc: 0.8360     
Epoch 97/100
8000/8000 [==============================] - 0s - loss: 0.3998 - acc: 0.8365      
Epoch 98/100
8000/8000 [==============================] - 0s - loss: 0.3997 - acc: 0.8350      
Epoch 99/100
8000/8000 [==============================] - 0s - loss: 0.4000 - acc: 0.8346     
Epoch 100/100
8000/8000 [==============================] - 0s - loss: 0.3998 - acc: 0.8375     
```
## Accuracy : 83.75%

