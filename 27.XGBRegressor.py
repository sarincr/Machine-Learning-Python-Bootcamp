import numpy as np
import pandas as pd
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
dataset = pd.read_csv("LinearData.csv")
X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
train_X, test_X, train_y, test_y = train_test_split(X, y,test_size = 0.25, random_state = 123)
xgb_r = xg.XGBRegressor(objective ='reg:linear', n_estimators = 15, seed = 123)
xgb_r.fit(train_X, train_y)
pred = xgb_r.predict(test_X)
print("RMSE : % f" ,np.sqrt(MSE(test_y, pred)))
