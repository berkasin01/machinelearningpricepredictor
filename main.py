import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("AMD.csv")
clean_data = data[["Open", "Close", "Adj Close", "Volume"]]

#regular linear regression model
# X = clean_data.iloc[:, 1:].values
# y = clean_data.iloc[:, 0].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=14)
# reg_mod = LinearRegression()
# reg_mod.fit(X_train, y_train)
#
# get_close = X_train[:, 0]
#
# plt.scatter(get_close, y_train, color="red")
# plt.plot(get_close, reg_mod.predict(X_train), color="blue")
# plt.show(block=True)

X = np.array(clean_data.Close)
y = np.array(clean_data.Open)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


print(y)


poly_reg = PolynomialFeatures(degree=10)
X_poly = poly_reg.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

plt.scatter(X, y, color="red")
plt.plot(X, poly_model.predict(X_poly))
plt.show(block=True)
