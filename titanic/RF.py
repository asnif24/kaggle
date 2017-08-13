#import matplotlib.pyplot as plt
#import pylab
import math

import pandas as pd
#import numpy as np
#import random as rnd

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import ensemble

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

# mean_Fare_train=train_df["Fare"].mean()

# print train_df.describe()
# print test_df.describe()


def normalize_data(data):
	data_mean=data.mean()
	for i in range(0,len(data.values)):
		data.values[i]=(data.values[i]-data_mean)/data_mean

def tran_nan_to_mean(data):
	data_mean=data.mean()
	for i in range(0,len(data.values)):
		if math.isnan(data.values[i]):
			data.values[i]=data_mean



#NaN to mean
tran_nan_to_mean(train_df["Age"])
tran_nan_to_mean(test_df["Age"])
tran_nan_to_mean(test_df["Fare"])

# #Normalize data
# # #PassengerId
# # normalize_data(train_df["PassengerId"])
# # normalize_data(test_df["PassengerId"])
# #Pclass
# normalize_data(train_df["Pclass"])
# normalize_data(test_df["Pclass"])
# #Age
# normalize_data(train_df["Age"])
# normalize_data(test_df["Age"])
# #SibSp
# normalize_data(train_df["SibSp"])
# normalize_data(test_df["SibSp"])
# #Parch
# normalize_data(train_df["Parch"])
# normalize_data(test_df["Parch"])
# #Fare
# normalize_data(train_df["Fare"])
# normalize_data(test_df["Fare"])


# #deal with FARE NaN
# for i in range(0,len(test_df["Fare"].values)):
# 	if math.isnan(test_df["Fare"].values[i]):
# 		test_df["Fare"].values[i]=test_df["Fare"].mean()

# mean_Fare_test=test_df["Fare"].mean()
# for i in range(0,len(test_df["Fare"].values)):
# 	(test_df["Fare"].values[i]-mean_Fare_test)/mean_Fare_test

# #KNN
# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# # predict=neigh.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
# print neigh.score(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])

# #SVC
# clf = svm.SVC()
# clf.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# # predict=clf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
# print clf.score(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])

# #LinearSVC
# lin_clf = svm.LinearSVC()
# lin_clf.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# # predict=lin_clf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
# print lin_clf.score(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])


train_tr = train_df.sample(860)
train_val = train_df.loc[~train_df.index.isin(train_tr.index)]

# RandomForestClassifier
# for trees in range(50,100,5):
# 	for layers in range(2,20,2):
# 		rdf = ensemble.RandomForestClassifier(n_estimators=trees,max_depth=layers)
# 		rdf.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# 		# predict=rdf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
# 		print "trees:", trees , " ;layers:", layers
# 		print rdf.score(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])

# # RandomForestClassifier
# for trees in range(50,350,50):
# 	for layers in range(5,30,5):
# 		rdf = ensemble.RandomForestClassifier(n_estimators=trees,max_depth=layers)
# 		rdf.fit(train_tr[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_tr["Survived"])
# 		# predict=rdf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
# 		print "trees:", trees , " ;layers:", layers , rdf.score(train_val[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_val["Survived"])

# # RandomForestClassifier
# for trees in range(50,350,50):
# 	for layers in range(4,15,2):
# 		rdf = ensemble.RandomForestClassifier(n_estimators=trees,max_depth=layers)
# 		rdf.fit(train_tr[["PassengerId","Sex","Pclass","Age","SibSp","Parch","Fare"]], train_tr["Survived"])
# 		# predict=rdf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
# 		print "trees:", trees , " ;layers:", layers , "train:", rdf.score(train_tr[["PassengerId","Sex","Pclass","Age","SibSp","Parch","Fare"]], train_tr["Survived"]) , "validation:", rdf.score(train_val[["PassengerId","Sex","Pclass","Age","SibSp","Parch","Fare"]], train_val["Survived"])


# rdf = ensemble.RandomForestClassifier(n_estimators=60,max_depth=4)
# rdf.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# predict=rdf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])

rdf = ensemble.RandomForestClassifier(n_estimators=150,max_depth=6)
rdf.fit(train_df[["PassengerId","Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
predict=rdf.predict(test_df[["PassengerId","Sex","Pclass","Age","SibSp","Parch","Fare"]])

# print "PassengerId"+","+"Survived"
# for k in range(892,1310):
# 	print str(k)+","+str(predict[k-892])

# print train_df.describe()
# print test_df.describe()

print "PassengerId"+","+"Survived"
for k in range(892,1310):
	print str(k)+","+str(predict[k-892])



