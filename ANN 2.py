# Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Importing the dataset
dataset = pd.read_csv("C:\\Users\\singh\\Desktop\\Sample data\\Churn_Modelling.csv")
print(dataset.head())
X = dataset.iloc[:, 3:-1].values  # Independent variables
Y = dataset.iloc[:, -1].values    # Dependent variable

# Encoding categorical data for 'Geography' column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Encoding 'Gender' column
le = LabelEncoder()
X[:, 4] = le.fit_transform(X[:, 4])

# Splitting dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building ANN
model = Sequential()

# Add the input layer and the first hidden layer
model.add(Dense(units=6, activation='relu', input_dim=X_train.shape[1]))

# Add the second hidden layer
model.add(Dense(units=6, activation='relu'))

# Add the output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compiling ANN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN on the Training Set
model.fit(X_train, y_train, batch_size=32, epochs=100)

# Evaluate the Model on the Test Set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Making Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(f"Predictions:\n{y_pred}")

#  printing a specefic customers
input_data = np.array([[1, 0, 0, 600, 1, 40, 3, 60000, 2,  50000]])

# Apply the same transformations
input_data_encoded = ct.transform(input_data)  # Apply OneHotEncoding to geographical data
input_data_encoded[:, 4] = le.transform(input_data_encoded[:, 4])  # Apply LabelEncoding to gender
input_data_scaled = sc.transform(input_data_encoded)  # Apply feature scaling

# Predict
prediction = model.predict(input_data_scaled)
print(prediction)
