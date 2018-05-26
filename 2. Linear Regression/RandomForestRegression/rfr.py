import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet = pd.read_csv('Position_Salaries.csv')
lvl = dataSet.iloc[:, 1:2].values
slry = dataSet.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators=100, random_state=0)
regression.fit(lvl, slry)

print(regression.predict(6.5))

fine_lvl = np.arange(min(lvl), max(lvl), .01)
fine_lvl = fine_lvl.reshape((len(fine_lvl), 1))

plt.scatter(lvl, slry, color='red')
plt.plot(fine_lvl, regression.predict(fine_lvl), color='green')
plt.title('Random Forest Regression')
plt.show()
