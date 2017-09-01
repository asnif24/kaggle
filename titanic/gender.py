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
		train_df["Sex"].values[i]=0
	else:
		train_df["Sex"].values[i]=1

for i in range(0,len(test_df["Sex"].values)):
	if test_df["Sex"].values[i]=="male":
		test_df["Sex"].values[i]=0
	else:
		test_df["Sex"].values[i]=1


print "PassengerId"+","+"Survived"
for k in range(892,1310):
	print str(k)+","+str(test_df["Sex"][k-892])





