# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# Importing the dataset
dataset = pd.read_csv('House_price.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,8].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 0:5])
X[:, 0:5]=missingvalues.transform(X[:, 0:5])

# Encoding categorical data
# Encoding the Independent Variable

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [5,6])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

X = np.array(ct.fit_transform(X), dtype=np.float)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)



# Feature Scaling
# WE  don't need to this section becase python librares is going to take care of it. ELYAR
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

#fitting Multiple linear regression to the Training set.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Predicting the Test set results
Y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination .
import statsmodels.api as sm
X =np.append(arr = np.ones((932,1)).astype(int),values =X ,axis=1)
X_opt = X[: , [0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = Y,exog =X_opt).fit()
regressor_OLS.summary()


X_opt = X[: , [0,1,2,3,4,5,6,7,8,9,10,11,12]]
regressor_OLS = sm.OLS(endog = Y,exog =X_opt).fit()
regressor_OLS.summary()


X_opt = X[: , [0,1,2,3,4,5,6,7,8,10,11,12]]
regressor_OLS = sm.OLS(endog = Y,exog =X_opt).fit()
regressor_OLS.summary()





















