from numpy import array
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import predict
from predict import predict
from sklearn.svm import LinearSVC, SVC
#from predict import SVM
from collections import defaultdict
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def plot(val):
	k_fold=[]
	sample=defaultdict(list)
	for k, N in val.items():
		k_fold.append(k)
		for a,v in N.items():
			#print(a,v)
			#plt.plot(k, v, label="line 2")
			#plt.plot('x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue',linewidth=4)
			#plt.plot('x', 'y2', data=df, marker='', color='olive', linewidth=2)
			#plt.plot('x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
			#plt.legend()
			if a==10:
				sample[a].append(v)
			elif a== 20:
				sample[a].append(v)
			elif a== 30:
				sample[a].append(v)
			elif a == 40:
				sample[a].append(v)

	plt.xlabel("Fold Number")
	plt.ylabel("Average Accuracy")
	plt.plot([1,2,3,4,5], sample[10], linestyle='solid', label="10 samples")
	plt.plot([1, 2, 3, 4, 5], sample[20], linestyle='solid', label="20 samples")
	plt.plot([1, 2, 3, 4, 5], sample[30], linestyle='solid', label="30 samples")
	plt.plot([1, 2, 3, 4, 5], sample[40], linestyle='solid', label="40 samples")
	plt.legend()
	plt.show()


def plot2(val1,val2):
	k_fold=[]
	acc=[]
	sample=defaultdict(list)
	for k, N in val1.items():
		k_fold.append(k)
		for a,v in N.items():
			#print(a,v)
			#plt.plot(k, v, label="line 2")
			#plt.plot('x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue',linewidth=4)
			#plt.plot('x', 'y2', data=df, marker='', color='olive', linewidth=2)
			#plt.plot('x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
			#plt.legend()
			if a==10:
				sample[a].append(v)
			elif a== 20:
				sample[a].append(v)
			elif a== 30:
				sample[a].append(v)
			elif a == 40:
				sample[a].append(v)
	for k, v in val2.items():
		acc.append(v)

	plt.xlabel("Fold Number")
	plt.ylabel("Average Accuracy")
	plt.plot([1,2,3,4,5], sample[40], linestyle='solid', label="Active Learning(SVM)")
	plt.plot([1, 2, 3, 4, 5], acc, linestyle='solid', label="Passive Learning(Logistic Regression)")

	plt.legend()
	plt.show()

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
scores=defaultdict(dict)
LRscores=defaultdict(dict)
fold_count = 0

for train, test in kf.split(X):
	fold_count+=1
	print("Fold: ",str(fold_count))
	#print('train: %s, test: %s' % (X[train], X[test]))
	X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

	for i in N:
		#X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
		print("Number of uncertain samples taken from pool:" + str(i))
		prdct = predict(i, X_train, y_train, X_test, y_test,'SVM')
		scores[fold_count][i]=prdct.Train()
		print("Test Accuracy: " + str(scores[fold_count][i]))

#plot(scores)
	LRprdct = predict(0, X_train, y_train, X_test, y_test, 'LR')
	LRscores[fold_count] = LRprdct.LRTrain()
#print(LRscores)
plot2(scores,LRscores)


