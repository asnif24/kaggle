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



train_df["Embarked_S"]=train_df["Embarked"].apply(lambda x : 1 if x=="S" else 0)
train_df["Embarked_C"]=train_df["Embarked"].apply(lambda x : 1 if x=="C" else 0)
train_df["Embarked_Q"]=train_df["Embarked"].apply(lambda x : 1 if x=="Q" else 0)

test_df["Embarked_S"]=test_df["Embarked"].apply(lambda x : 1 if x=="S" else 0)
test_df["Embarked_C"]=test_df["Embarked"].apply(lambda x : 1 if x=="C" else 0)
test_df["Embarked_Q"]=test_df["Embarked"].apply(lambda x : 1 if x=="Q" else 0)

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


train_tr = train_df.sample(800)
train_val = train_df.loc[~train_df.index.isin(train_tr.index)]


# #KNN
# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# # predict=neigh.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
# print neigh.score(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])

# #KNN
# for k in range(1,9,1):
# 	neigh = KNeighborsClassifier(n_neighbors=k)
# 	neigh.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]], train_df["Survived"])
# 	# predict=neigh.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
# 	# print neigh.score(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# 	print "K:", k , "train:", neigh.score(train_tr[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]], train_tr["Survived"]) , "validation:", neigh.score(train_val[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]], train_val["Survived"])



# #SVC
# clf = svm.SVC()
# clf.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]], train_df["Survived"])
# # predict=clf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
# # print clf.score(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# print "SVC:", "SVC" , "train:", clf.score(train_tr[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]], train_tr["Survived"]) , "validation:", clf.score(train_val[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]], train_val["Survived"])

# #LinearSVC
# lin_clf = svm.LinearSVC()
# lin_clf.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# # predict=lin_clf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
# print lin_clf.score(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])


# # RandomForestClassifier
# for trees in range(50,250,20):
# 	for layers in range(4,18,2):
# 		rdf = ensemble.RandomForestClassifier(n_estimators=trees,max_depth=layers)
# 		rdf.fit(train_tr[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]], train_tr["Survived"])
# 		# predict=rdf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
# 		print "trees:", trees , " ;layers:", layers , "train:", rdf.score(train_tr[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]], train_tr["Survived"]) , "validation:", rdf.score(train_val[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]], train_val["Survived"])


# rdf = ensemble.RandomForestClassifier(n_estimators=60,max_depth=4)
# rdf.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# predict=rdf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])

# rdf = ensemble.RandomForestClassifier(n_estimators=150,max_depth=6)
# rdf.fit(train_df[["PassengerId","Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# predict=rdf.predict(test_df[["PassengerId","Sex","Pclass","Age","SibSp","Parch","Fare"]])

# rdf = ensemble.RandomForestClassifier(n_estimators=210,max_depth=10)
# rdf.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]], train_df["Survived"])
# predict=rdf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]])

# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]], train_df["Survived"])
# predict=neigh.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]])

clf = svm.SVC()
clf.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]], train_df["Survived"])
predict=clf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q"]])


# print "PassengerId"+","+"Survived"
# for k in range(892,1310):
# 	print str(k)+","+str(predict[k-892])

# print train_df.describe()
# print test_df.describe()

print "PassengerId"+","+"Survived"
for k in range(892,1310):
	print str(k)+","+str(predict[k-892])



