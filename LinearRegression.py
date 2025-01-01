import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Load the data

train_data = pd.read_csv('C:/Users/omars/OneDrive/Desktop/Work/Prodigy InfoTech/Task 1/train.csv')
test_data = pd.read_csv('C:/Users/omars/OneDrive/Desktop/Work/Prodigy InfoTech/Task 1/test.csv')

#print(train_data.head())
#print(train_data.info())
#print(train_data.describe())

# Adding new columns for total square foot and total number of rooms
train_data['TotalSquareFootage'] = train_data['1stFlrSF'] + train_data['2ndFlrSF'] + train_data['TotalBsmtSF']
train_data['TotalRooms'] = train_data['BsmtFullBath'] + train_data['BsmtHalfBath'] + train_data['FullBath'] + train_data['HalfBath'] + train_data['BedroomAbvGr']

test_data['TotalSquareFootage'] = test_data['1stFlrSF'] + test_data['2ndFlrSF'] + test_data['TotalBsmtSF']
test_data['TotalRooms'] = test_data['BsmtFullBath'] + test_data['BsmtHalfBath'] + test_data['FullBath'] + test_data['HalfBath'] + test_data['BedroomAbvGr']


#print(train_data[['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'TotalSquareFootage']].head())
#print(train_data[['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotalRooms']].head())
#print(train_data[['SalePrice']].head())

# Calculating parameters from training data
X_train = train_data[['TotalSquareFootage', 'TotalRooms']]
X_train_scaled = scaler.fit_transform(X_train)

# Fill NaN with zero in test data
test_data['TotalSquareFootage'] = test_data['TotalSquareFootage'].fillna(0)  
test_data['TotalRooms'] = test_data['TotalRooms'].fillna(0)

# Apply same fitting of train data to test data
X_test = test_data[['TotalSquareFootage', 'TotalRooms']]
X_test_scaled = scaler.transform(X_test)

Y_train = train_data['SalePrice']

# Initialization of Linear Regression model

model = LinearRegression()

# Linear Regression fit on scaled data
model.fit(X_train_scaled, Y_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Make price predictions for test data
test_predictions = model.predict(X_test_scaled)
test_data['PredictedSalePrice'] = test_predictions

print(train_data[['TotalSquareFootage', 'TotalRooms', 'SalePrice']].head())
print(test_data[['TotalSquareFootage', 'TotalRooms', 'PredictedSalePrice']].head())