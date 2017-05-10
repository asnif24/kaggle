#import matplotlib.pyplot as plt
#import pylab
import math

import pandas as pd
#import numpy as np
#import random as rnd

from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

for i in range(0,len(train_df["Sex"].values)):
	if train_df["Sex"].values[i]=="male":
		train_df["Sex"].values[i]=1.0
	else:
		train_df["Sex"].values[i]=0.0

for i in range(0,len(test_df["Sex"].values)):
	if test_df["Sex"].values[i]=="male":
		test_df["Sex"].values[i]=1.0
	else:
		test_df["Sex"].values[i]=0.0

mean_Fare_train=train_df["Fare"].mean()
for i in range(0,len(train_df["Fare"].values)):
	(train_df["Fare"].values[i]-mean_Fare_train)/mean_Fare_train

for i in range(0,len(test_df["Fare"].values)):
	if math.isnan(test_df["Fare"].values[i]):
		test_df["Fare"].values[i]=test_df["Fare"].mean()

mean_Fare_test=test_df["Fare"].mean()
for i in range(0,len(test_df["Fare"].values)):
	(test_df["Fare"].values[i]-mean_Fare_test)/mean_Fare_test

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_df[["Sex","Pclass","SibSp","Parch","Fare"]], train_df["Survived"])
predict=neigh.predict(test_df[["Sex","Pclass","SibSp","Parch","Fare"]])

print "PassengerId"+","+"Survived"
for k in range(892,1310):
	print str(k)+","+str(predict[k-892])