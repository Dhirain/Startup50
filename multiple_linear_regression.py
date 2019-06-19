# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

data1 = dataset.iloc[: , [0,1,2,4]].values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Plot
X_plot_rd = X_test[:,2]
X_plot_rd.sort()
X_plot_admin = X_test[:,3]
X_plot_admin.sort()
X_plot_market = X_test[:,4]
X_plot_market.sort()
y_plot = y_test
y_plot.sort()
y_pred_plot = y_pred
y_pred_plot.sort()

plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.plot(X_plot_rd,y_plot,color='red')
plt.plot(X_plot_rd,y_pred_plot,color='blue')


plt.xlabel('Administrator Spend')
plt.ylabel('Profit')
plt.plot(X_plot_admin,y_plot,color='red')
plt.plot(X_plot_admin,y_pred_plot,color='blue')


plt.xlabel('Market Spend')
plt.ylabel('Profit')
plt.plot(X_plot_market,y_plot,color='red')
plt.plot(X_plot_market,y_pred_plot,color='blue')
