import csv
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

ftrain=open("train.csv","r")
csvtrain=csv.reader(ftrain)
ltrain=list(csvtrain)

ftest=open("test.csv","r")
csvtest=csv.reader(ftest)
ltest=list(csvtest)

df=pd.DataFrame(ltrain[1:len(ltrain)], columns=ltrain[0])

#df[df["Sex"]=="male"]["Sex"]=0
#df[df["Sex"]=="female"]["Sex"]=1
#df["Sex"]=1
#print df.head(15)
#print df.head(10)
print df[["Survived","Pclass"]].plot()



#print test_info
#neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(info, result)
#predict=neigh.predict(test_info)


ftrain.close()
ftest.close()
