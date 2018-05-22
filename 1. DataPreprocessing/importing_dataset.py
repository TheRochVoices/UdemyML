import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data
dataset = pd.read_csv('Data.csv')
ind = dataset.iloc[:, 0:3].values  #getting independednt variables
dep = dataset.iloc[:, 3].values		#getting dependent variabbles in seprate vector

#handling missing values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0)
ind[:, 1:3] = imputer.fit_transform(ind[:, 1:3])


#handling categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
dep = labelencoder_y.fit_transform(dep)
print(dep)

#Important : First convert the features into int values for onehotencoding using LabelEncoder
labelencoder_x = LabelEncoder()
ind[:, 0] = labelencoder_x.fit_transform(ind[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
ind = onehotencoder.fit_transform(ind).toarray()

#splitting dataset into train and test sets

from sklearn.cross_validation import train_test_split
ind_train, ind_test, dep_train, dep_test = train_test_split(ind, dep, test_size = 0.2, random_state=0)

print(ind_train)
print("************************************************")
#scaling down the large valued columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
ind_train = scaler.fit_transform(ind_train)
ind_test = scaler.transform(ind_test)
print(ind_test)
