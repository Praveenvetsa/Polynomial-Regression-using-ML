# In this code we do not split the data like train and test data because we are using simple dataset containing only 10 records 
# so we cannot split the data we can split the data having more dataset containing many number of records 

import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\18th\1.POLYNOMIAL REGRESSION\emp_salary.csv')

X = data.iloc[:,1:2] # Independent variable

y = data.iloc[:,2] # Dependent variable

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
#poly_reg = PolynomialFeatures() #By default it takes 2
#poly_reg = PolynomialFeatures(degree = 3)
#poly_reg = PolynomialFeatures(degree = 4)
poly_reg = PolynomialFeatures(degree = 5)
#poly_reg = PolynomialFeatures(degree = 6)

X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualization for linaer regression
plt.scatter(X,y, color ='red')
plt.plot(X, lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualization for polynomial regression
plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'Blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Prediction
lin_reg.predict([[6.5]])

lin_reg2.predict(poly_reg.fit_transform([[6.5]]))

lin_reg2.predict(poly_reg.fit_transform([[20]]))





