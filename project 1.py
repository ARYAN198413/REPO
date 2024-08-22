import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# Load dataset
dataset = pd.read_csv("C:\\Users\\singh\\Downloads\\CAR DETAILS FROM CAR DEKHO.csv")

# Display basic information
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.isnull().sum())

# Encode categorical data
dataset.replace({'fuel': {'Petrol': 0, 'Diesel': 1, 'LPG': 2, 'CNG': 3, 'Electric': 4}}, inplace=True)
dataset.replace({'seller_type': {'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2}}, inplace=True)
dataset.replace({'transmission': {'Manual': 0, 'Automatic': 1, 'Electric': 2, 'Test Drive Car': 3}}, inplace=True)
dataset.replace({'owner': {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3}}, inplace=True)

print(dataset.head())

X = dataset.drop(['name', 'selling_price'], axis=1)
Y = dataset['selling_price']

print(X.head())
print(Y.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict on the training and test sets
y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

# Evaluate Linear Regression model
print("Linear Regression - Training Set Evaluation:")
print("R-squared:", metrics.r2_score(y_train, y_train_pred))
print("Mean Absolute Error:", metrics.mean_absolute_error(y_train, y_train_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_train, y_train_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

print("\nLinear Regression - Test Set Evaluation:")
print("R-squared:", metrics.r2_score(y_test, y_test_pred))
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_test_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_test_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

# Plot Predictions vs Actual Values
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.5)
plt.title('Training Set: Actual vs Predicted')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, color='green', alpha=0.5)
plt.title('Test Set: Actual vs Predicted')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')

plt.tight_layout()
plt.show()

# Train Lasso Regression model
lasso_reg = Lasso(alpha=0.1)  # You can tune alpha
lasso_reg.fit(X_train, y_train)

# Predict on the training and test sets using Lasso
y_train_pred_lasso = lasso_reg.predict(X_train)
y_test_pred_lasso = lasso_reg.predict(X_test)

# Evaluate Lasso Regression model
print("\nLasso Regression - Training Set Evaluation:")
print("R-squared:", metrics.r2_score(y_train, y_train_pred_lasso))
print("Mean Absolute Error:", metrics.mean_absolute_error(y_train, y_train_pred_lasso))
print("Mean Squared Error:", metrics.mean_squared_error(y_train, y_train_pred_lasso))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_train, y_train_pred_lasso)))

print("\nLasso Regression - Test Set Evaluation:")
print("R-squared:", metrics.r2_score(y_test, y_test_pred_lasso))
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_test_pred_lasso))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_test_pred_lasso))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_lasso)))

# Compare Linear and Lasso Regression
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, color='green', alpha=0.5, label='Linear Regression')
plt.title('Test Set: Linear Regression Predictions')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred_lasso, color='red', alpha=0.5, label='Lasso Regression')
plt.title('Test Set: Lasso Regression Predictions')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.legend()

plt.tight_layout()
plt.show()