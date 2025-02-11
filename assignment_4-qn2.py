import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("weight-height.csv")


plt.figure()
plt.scatter(data["Height"], data["Weight"], alpha=0.5)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Scatter Plot of Height vs Weight")
plt.show()


X = data[["Height"]]
y = data["Weight"]


model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)


plt.figure()
plt.scatter(X, y, alpha=0.5, label="Actual data")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Linear Regression: Height vs Weight")
plt.legend()
plt.show()


rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")

