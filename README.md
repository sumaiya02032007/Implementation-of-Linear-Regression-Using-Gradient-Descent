# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and select features.

2.Split data into training and testing sets.

3.Scale the input features.

4.Train the multi-output regression model.

5.Predict outputs and calculate error.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: S.SUMAIYA
RegisterNumber:  21225040437
*/

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

data = fetch_california_housing()

x = data.data[:, :3]
y = np.c_[data.target, data.data[:, 6]]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
y_train = scaler.fit_transform(y_train)

model = MultiOutputRegressor(SGDRegressor(random_state = 42))
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)

print(f"Mean Squared Error (MSE) : {mse}")
print("Sample Prediction (House Price,Population) :",)
print(y_pred[:5])

```

## Output:
<img width="749" height="231" alt="image" src="https://github.com/user-attachments/assets/3cfe8c1e-a7ca-43dd-bb7b-2ba5a824bebe" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
