# Part 1: Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Encoding categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encoding geography
labelencoder_x_1 = LabelEncoder()
x[:,1]= labelencoder_x_1.fit_transform(x[:, 1])

# Encoding gender
labelencoder_x_2 = LabelEncoder()
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()

# avoid dummy variable trap
x = x[:,1:]


# Splitting the dataset into train and test set 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test  = sc.fit_transform(x_test)

## Part II: Making ANN

# Importing the Keras package
import keras
from keras.models import Sequential # to initiate neural network
from keras.layers import Dense  # to create layers
from keras.layers import Dropout  # for dropout regularization

# Initializing the ANN
classifier = Sequential()

# Adding layers
# output_dim = average(input and output nodes) = (11+1)/2 = 6

# Adding the input layer and first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform',activation='relu', input_dim = 11))
# Dense function is to add a fully connected layer

# With dropout (Dropout regulrization is to reduce overfitting if required)
# classifier.add(Dropout(p=0.1))

 
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform',activation='relu'))
# With dropout
# classifier.add(Dropout(p=0.1))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform',activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting ANN to train set
classifier.fit(x_train, y_train, batch_size=10, nb_epoch = 100)


## Making Predictions and Evaluating the Model
# Predicting the test set results:
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5) # Cutoff is set at 0.5

# Building the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Checking the Accuracy
accuracy = (cm[0, 0] + cm[1, 1])/(cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0])
accuracy


##### New test case #####

""" Predict if the following customer will churn/leave the bank:
Geography : France
Credit Score : 600
Gender: Male
Age : 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active member: Yes
Estimated Salary : 50000 """

new_pred = classifier.predict(sc.transform([[0,0,600,1,40,3,60000,2,1,1,50000]]))
new_pred = (new_pred>0.5)
new_pred

# Further refining the evaluation using K-fold cross-validation.
""" Since, K -fold cross validation is in scikit learn we will have to 
establish a relation between both for which we use below function"""

from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform',activation='relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform',activation='relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn= build_classifier, batch_size = 10, nb_epoch = 100)
# Computing accuracies of 10 folds
accuracies = cross_val_score(estimator=classifier, X = x_train, y= y_train, cv=3, n_jobs= -1) 

mean = accuracies.mean() # Computing means of accuracies of 10 folds
variance = accuracies.std()


# Tuning the ANN (Hyper parameter Tuning)
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform',activation='relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform',activation='relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform',activation='sigmoid'))
    classifier.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25,32],
              'nb_epoch': [100,500],
              'optimizer':['adam','rmsprop']}
# grid search on parameters
grid_search = GridSearchCV(estimator=classifier, 
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)
grid_search = grid_search.fit(x_train, y_train)


best_parameters = grid_search.best_params_ # best parameters
best_accuracy = grid_search.best_score_ # best accuracy

# print values
print("Best parameters: {best_params}".format(best_params=best_parameters))
print("Best accuracy: {best_acc}".format(best_acc=best_accuracy))

""" Best parameters on running the model were:
    Batch_size: 25
    nb_epoch : 100
    optimizer: 'adam' 
    
    Best Accuracy = 83.75% """
    
    


