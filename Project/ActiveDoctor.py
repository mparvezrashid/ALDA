from numpy import array
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import predict
from predict import predict


filename = "heart.csv"
data = pd.read_csv(filename)
#X = np.loadtxt(filename, delimiter=",", skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
#y = np.loadtxt(filename, delimiter=",", skiprows=1,usecols=(13))
#y=data.target
#X= data.drop('target', axis=1)

X = data.iloc[:, 0: 13].to_numpy()
y = data.iloc[:, 13].to_numpy()

sclr = MinMaxScaler(feature_range=(0, 1))
X = sclr.fit_transform(X)
kf = KFold(5, True, 1)
N=[10,20,30,40]
scores=[]
fold_count = 0
for train, test in kf.split(X):
	fold_count+=1
	print("Fold: ",str(fold_count))
	#print('train: %s, test: %s' % (X[train], X[test]))
	X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

	for i in N:
		print("Number of uncertain samples taken from pool:" + str(i))
		prdct = predict(i, X_train, y_train, X_test, y_test)
		prdct.Train()


