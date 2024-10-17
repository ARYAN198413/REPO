# Medical insurance cost prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("C:\\Users\\singh\\Downloads\\insurance.csv")
print(dataset.head())
# Info about data
print(dataset.shape)
print(dataset.info())
print(dataset.describe())
# Gender distribution
plt.figure(figsize = (6,6))
sns.countplot(x = 'sex', data = dataset)
plt.title('Sex distribution')
plt.show()

# #  bmi distribution
plt.figure(figsize = (6,6))
sns.distplot(dataset['bmi'])
plt.title("Age distribution")
plt.show()

# # smoker column
plt.figure(figsize = (6,6))
sns.countplot(x = 'smoker', data = dataset)
plt.title('Smoker distribution')
plt.show()

# region column
plt.figure(figsize = (6,6))
sns.countplot(x = 'region', data = dataset)
plt.title('region distribution')
plt.show()

#  encoding categorical data
print(dataset.replace({'sex' : {'male' : 0, 'female' : 1}}, inplace = True))
print(dataset.head())

print(dataset.replace({'smoker' : {'yes' : 0, 'no' : 1}}, inplace = True))
print(dataset.head())

print(dataset.replace({'region' : {'southwest' : 0, 'southeast' : 1, 'northwest' : 2, 'northeast' : 3}}, inplace = True))
print(dataset.head())

X = dataset.drop(columns='charges') 
Y = dataset['charges']
print(X)

#  splitting dataset

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# model training

regressor = LinearRegression()

regressor.fit(X_train, Y_train)
#  prediction on training data
a = training_data_prediction = regressor.predict(X_train)

print(a)

#  R squared value
b = r2_train = metrics.r2_score(Y_train, training_data_prediction )
print(b)

#  prediction on test data
c = test_data_prediction = regressor.predict(X_test)
print(c)

r2_test = metrics.r2_score(Y_test, test_data_prediction )
print(r2_test)

input_data = (31,1,25.74,0,1,0)
input_data_as_array = np.asarray(input_data)

input_data_reshaped = input_data_as_array.reshape(1,-1)
prediction = regressor.predict(input_data_reshaped)
print(prediction)







