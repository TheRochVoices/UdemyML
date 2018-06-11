
import pandas as pd
import numpy as np

df = pd.read_csv('Social_Network_Ads.csv')
df.sample(5)
x = df.iloc[:, 2:4].values
y = df.iloc[:, 4]

'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lE = LabelEncoder()
x[:,0] = lE.fit_transform(x[:, 0])'''

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

import keras
from keras.models import Sequential
from keras.layers import Dense

ann = Sequential()
ann.add(Dense(2, kernel_initializer = 'uniform', activation = 'relu', input_shape = (2,)))
ann.add(Dense(1, kernel_initializer = 'uniform', activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(x_test, y_test, batch_size=10, epochs=65)
# an accuracy of 88... While considering gender into consideration, the accuracy dropped to 68

y_pred = ann.predict(x_test)
y_pred = y_pred>.5

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

