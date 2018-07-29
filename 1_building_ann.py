
# part -1 DATA PRE-PROCESSING

# importing lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values 
y = dataset.iloc[:, -1].values

# Encoding categorical data into numeric 1 and 0
# two categories gender and geography acc to the dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# for the geography
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
# for the gender
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:] 

# train and test the datasets
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# part -2 BUILD ANN
import keras
from keras.models import Sequential
from keras.layers import Dense


#initailizing ANN
classifier = Sequential()

# ADDING I/P to first hidden layer 
# (rectifier func used for hidden layer and sigmoid function for the output layer)

# output_dim = inputs + outputs / 2 = 11 + 1 = 6 
# input_dim = 11 [as the columns taken from credit_score to exited]
# activation = relu --> rectifier method
classifier.add(Dense(units =6, kernel_initializer = 'uniform', activation ='relu', input_dim= 11))

# ADDING 2nd hidden layer
classifier.add(Dense(units =6, kernel_initializer = 'uniform', activation = 'relu'))

# ADDING OUTPUT LAYER
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# COMPILING ANN
# loss = 'binary_crossentropy'     ---> if your dependent variable has a binary outcome 
# loss = 'categorial_crossentropy' ---> if your dependent variable has more than two outcomes like three categories
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics =['accuracy'])

# FIT THE ANN to training set
classifier.fit(x_train, y_train, batch_size=10,nb_epoch=100)

# MAKING PREDICTION
# predicting the test set results 
y_pred = classifier.predict(x_test)
# [ returns probability in TRUE OR FALSE of customers leaving bank]
y_pred = (y_pred > 0.5)

# MAKING CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
