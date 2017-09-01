import math
import pandas as pd
from sklearn import ensemble
from sklearn import svm

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Sex to 0 or 1
train_df["Sex"][train_df.Sex == "male"] = 0
train_df["Sex"][train_df.Sex == "female"] = 1
test_df["Sex"][test_df.Sex == "male"] = 0
test_df["Sex"][test_df.Sex == "female"] = 1

# Embarked
train_df["Embarked_S"]=train_df["Embarked"].apply(lambda x : 1 if x=="S" else 0)
train_df["Embarked_C"]=train_df["Embarked"].apply(lambda x : 1 if x=="C" else 0)
train_df["Embarked_Q"]=train_df["Embarked"].apply(lambda x : 1 if x=="Q" else 0)
test_df["Embarked_S"]=test_df["Embarked"].apply(lambda x : 1 if x=="S" else 0)
test_df["Embarked_C"]=test_df["Embarked"].apply(lambda x : 1 if x=="C" else 0)
test_df["Embarked_Q"]=test_df["Embarked"].apply(lambda x : 1 if x=="Q" else 0)

# Name
# train_df["Names"].unique() get unique items
dic={'Mr':1, 'Mrs':2, 'Miss':3, 'Master':4, 'Don':5, 'Rev':6, 'Dr':7, 'Mme':8, 'Ms':9, 'Major':10, 'Lady':11, 'Sir':12, 'Mlle':13, 'Col':14, 'Capt':15, 'Countess':16, 'Jonkheer':17, 'Dona':18}
train_df['Names'] = train_df['Name'].apply(lambda x: dic[x.split(".")[0].split()[-1]])
test_df['Names'] = test_df['Name'].apply(lambda x: dic[x.split(".")[0].split()[-1]])
train_df["Name_cnt"]=train_df["Name"].apply(lambda x: len(x))
test_df["Name_cnt"]=test_df["Name"].apply(lambda x: len(x))


# Cabin
cabin_dic={'N':9, 'C':3, 'E':5, 'G':7, 'D':4, 'A':1, 'B':2, 'F':6, 'T':8}
train_df["Cabin_1"]=train_df["Cabin"].fillna("N").apply(lambda x: cabin_dic[x[0]])
test_df["Cabin_1"]=test_df["Cabin"].fillna("N").apply(lambda x: cabin_dic[x[0]])



# dealing with Age NaN
rfr_age=ensemble.RandomForestRegressor(n_estimators=200,max_depth=20)
rfr_age.fit(train_df[["Sex","Pclass","SibSp","Parch","Fare","Names"]][train_df.Age.isnull()==False], train_df["Age"][train_df.Age.isnull()==False])
train_df["Age"][train_df.Age.isnull()==True] = rfr_age.predict(train_df[["Sex","Pclass","SibSp","Parch","Fare","Names"]][train_df.Age.isnull()==True])
test_df["Age"][test_df.Age.isnull()==True] = rfr_age.predict(test_df[["Sex","Pclass","SibSp","Parch","Fare","Names"]][test_df.Age.isnull()==True])


# test fare nan to mean
test_df.Fare[test_df.Fare.isnull()==True]=test_df.Fare.mean()

# parameters
parameters_tr = ["Sex","Pclass","Age","SibSp","Parch","Fare","Embarked_S","Embarked_C","Embarked_Q","Names"]




# cross val
train_tr = train_df.sample(850)
train_val = train_df.loc[~train_df.index.isin(train_tr.index)]

# RandomForestClassifier
for trees in range(50,200,20):
	for layers in range(4,30,4):
		rdf = ensemble.RandomForestClassifier(n_estimators=trees,max_depth=layers,criterion='gini')
		rdf.fit(train_tr[parameters_tr], train_tr["Survived"])
		print "trees:", trees , " ;layers:", layers , "train:", rdf.score(train_tr[parameters_tr], train_tr["Survived"]) , "validation:", rdf.score(train_val[parameters_tr], train_val["Survived"])


# rdf = ensemble.RandomForestClassifier(n_estimators=200,max_depth=10,criterion='gini')
# rdf.fit(train_df[parameters_tr], train_df["Survived"])
# predict=rdf.predict(test_df[parameters_tr])



print "PassengerId"+","+"Survived"
for k in range(892,1310):
	print str(k)+","+str(predict[k-892])





