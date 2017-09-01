#import matplotlib.pyplot as plt
#import pylab
import math

import pandas as pd
#import numpy as np
#import random as rnd

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import ensemble

from sklearn.model_selection import cross_val_score

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

train_df["Cabin_pre"]=train_df["Cabin"].fillna(value="N").apply(lambda x :str(x)[0])
train_df["Cabin_A"]=train_df["Cabin_pre"].apply(lambda x : 1 if x=="A" else 0)
train_df["Cabin_B"]=train_df["Cabin_pre"].apply(lambda x : 1 if x=="B" else 0)
train_df["Cabin_C"]=train_df["Cabin_pre"].apply(lambda x : 1 if x=="C" else 0)
train_df["Cabin_D"]=train_df["Cabin_pre"].apply(lambda x : 1 if x=="D" else 0)
train_df["Cabin_E"]=train_df["Cabin_pre"].apply(lambda x : 1 if x=="E" else 0)
train_df["Cabin_F"]=train_df["Cabin_pre"].apply(lambda x : 1 if x=="F" else 0)
train_df["Cabin_G"]=train_df["Cabin_pre"].apply(lambda x : 1 if x=="G" else 0)
train_df["Cabin_T"]=train_df["Cabin_pre"].apply(lambda x : 1 if x=="T" else 0)


test_df["Embarked_S"]=test_df["Embarked"].apply(lambda x : 1 if x=="S" else 0)
test_df["Embarked_C"]=test_df["Embarked"].apply(lambda x : 1 if x=="C" else 0)
test_df["Embarked_Q"]=test_df["Embarked"].apply(lambda x : 1 if x=="Q" else 0)

test_df["Cabin_pre"]=test_df["Cabin"].fillna(value="N").apply(lambda x :str(x)[0])
test_df["Cabin_A"]=test_df["Cabin_pre"].apply(lambda x : 1 if x=="A" else 0)
test_df["Cabin_B"]=test_df["Cabin_pre"].apply(lambda x : 1 if x=="B" else 0)
test_df["Cabin_C"]=test_df["Cabin_pre"].apply(lambda x : 1 if x=="C" else 0)
test_df["Cabin_D"]=test_df["Cabin_pre"].apply(lambda x : 1 if x=="D" else 0)
test_df["Cabin_E"]=test_df["Cabin_pre"].apply(lambda x : 1 if x=="E" else 0)
test_df["Cabin_F"]=test_df["Cabin_pre"].apply(lambda x : 1 if x=="F" else 0)
test_df["Cabin_G"]=test_df["Cabin_pre"].apply(lambda x : 1 if x=="G" else 0)
test_df["Cabin_T"]=test_df["Cabin_pre"].apply(lambda x : 1 if x=="T" else 0)

#NaN to mean
# tran_nan_to_mean(train_df["Age"])

#deal with age nan
train_df["Age_0"]=train_df["Age"].fillna(value=0)
train_df["Age_nan"]=train_df["Age_0"].apply(lambda x : 1 if x==0 else 0)

tran_nan_to_mean(test_df["Age"])
tran_nan_to_mean(test_df["Fare"])

train_df["Age_1"]=train_df["Age"].fillna(value=-1).apply(lambda x: int(x/10) if x>0 else x)
rdf_age = ensemble.RandomForestClassifier(n_estimators=250,max_depth=6,criterion='gini')
rdf_age.fit(train_df[train_df["Age_1"]>0][["Sex","Pclass","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T","Survived"]], train_df[train_df["Age_1"]>0]["Age_1"])
age_predict=rdf_age.predict(train_df[train_df["Age_1"]<0][["Sex","Pclass","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T","Survived"]])
train_df.Age_1[train_df.Age_1<0]=age_predict

# for trees in range(50,200,20):
# 	for layers in range(4,30,4):
# 		rdf_age = ensemble.RandomForestClassifier(n_estimators=trees,max_depth=layers,criterion='gini')
# 		rdf_age.fit(train_df[train_df["Age_1"]>0][["Sex","Pclass","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T","Survived"]], train_df[train_df["Age_1"]>0]["Age_1"])
# 		age_predict=rdf_age.predict(train_df[train_df["Age_1"]<0][["Sex","Pclass","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T","Survived"]])
# 		train_df.Age_1[train_df.Age_1<0]=age_predict
# 		print "trees:", trees , " ;layers:", layers , "train:", rdf_age.score(train_tr[["Sex","Pclass","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T","Survived"]], train_tr["Age_1"]) , "validation:", rdf.score(train_val[["Sex","Pclass","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T","Survived"]], train_val["Age_1"])



test_df["Age_1"]=test_df["Age"].fillna(value=-1).apply(lambda x: int(x/10) if x>0 else x)
# rdf_age_test = ensemble.RandomForestClassifier(n_estimators=100,max_depth=10,criterion='gini')
# rdf_age_test.fit(test_df[test_df["Age_1"]>0][["Sex","Pclass","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]], test_df[test_df["Age_1"]>0]["Age_1"])
# test_df.Age_1[test_df.Age_1<0]=rdf_age_test.predict(test_df[test_df["Age_1"]<0][["Sex","Pclass","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]])

# svr_age = ensemble.RandomForestClassifier()
# svr_age.fit(train_df[train_df["Age_0"]>0][["Sex","Pclass","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T","Survived"]], train_df[train_df["Age_0"]>0]["Age_1"])
# train_df[train_df["Age_0"]<0].apply(lambda x: svr_age.predict(x[["Sex","Pclass","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T","Survived"]])[0])



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


train_tr = train_df.sample(850)
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

# RandomForestClassifier
for trees in range(50,200,20):
	for layers in range(4,30,4):
		rdf = ensemble.RandomForestClassifier(n_estimators=trees,max_depth=layers,criterion='gini')
		rdf.fit(train_tr[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_pre"]], train_tr["Survived"])
		# predict=rdf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
		print "trees:", trees , " ;layers:", layers , "train:", rdf.score(train_tr[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_pre"]], train_val["Survived"])
		# print "trees:", trees , " ;layers:", layers , "train:", cross_val_score(rdf,train_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q","Embarked_Q","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]], train_df["Survived"])

# # RandomForestClassifier
# for trees in range(50,200,20):
# 	for layers in range(4,30,4):
# 		rdf = ensemble.RandomForestClassifier(n_estimators=trees,max_depth=layers,criterion='gini')
# 		rdf.fit(train_tr[["Sex","Pclass","Age_1","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]], train_tr["Survived"])
# 		# predict=rdf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])
# 		print "trees:", trees , " ;layers:", layers , "train:", rdf.score(train_tr[["Sex","Pclass","Age_1","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]], train_tr["Survived"]) , "validation:", rdf.score(train_val[["Sex","Pclass","Age_1","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]], train_val["Survived"])
# 		# print "trees:", trees , " ;layers:", layers , "train:", cross_val_score(rdf,train_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q","Embarked_Q","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]], train_df["Survived"])


# rdf = ensemble.RandomForestClassifier(n_estimators=60,max_depth=4)
# rdf.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# predict=rdf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare"]])

# rdf = ensemble.RandomForestClassifier(n_estimators=150,max_depth=6)
# rdf.fit(train_df[["PassengerId","Sex","Pclass","Age","SibSp","Parch","Fare"]], train_df["Survived"])
# predict=rdf.predict(test_df[["PassengerId","Sex","Pclass","Age","SibSp","Parch","Fare"]])

# rdf = ensemble.RandomForestClassifier(n_estimators=200,max_depth=8)
# rdf.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]], train_df["Survived"])
# predict=rdf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]])

# rdf = ensemble.RandomForestClassifier(n_estimators=120,max_depth=20,criterion='gini')
# rdf.fit(train_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]], train_df["Survived"])
# predict=rdf.predict(test_df[["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]])

# rdf = ensemble.RandomForestClassifier(n_estimators=110,max_depth=12,criterion='gini')
# rdf.fit(train_df[["Sex","Pclass","Age_1","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]], train_df["Survived"])
# predict=rdf.predict(test_df[["Sex","Pclass","Age_1","SibSp","Parch","Fare","Embarked_S","Embarked_C","Cabin_A","Cabin_B","Cabin_C","Cabin_D","Cabin_E","Cabin_F","Cabin_G","Cabin_T"]])


# print "PassengerId"+","+"Survived"
# for k in range(892,1310):
# 	print str(k)+","+str(predict[k-892])

# print train_df.describe()
# print test_df.describe()

print "PassengerId"+","+"Survived"
for k in range(892,1310):
	print str(k)+","+str(predict[k-892])



