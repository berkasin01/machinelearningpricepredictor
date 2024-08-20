import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("AMD.csv")
no_date_data = data.drop("Date", axis=1)

data_pick = "Open"
X = no_date_data.drop(data_pick, axis=1)
y = no_date_data[data_pick]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=10)

reg_obj = LinearRegression()

reg_obj.fit(X_train, y_train)

y_predict = reg_obj.predict(X_test)

mse = mean_squared_error(y_test, y_predict)

predict_values = reg_obj.predict(X_train)
residuals = (predict_values - y_train)

obj_mse = reg_obj.score(X_train, y_train)

df = pd.DataFrame({"Actual": y_test, "Predicted": y_predict})

print(df)
fig = sns.scatterplot(x='Actual', y='Predicted', data=df)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show(block=True)
