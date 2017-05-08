import csv
from sklearn.neighbors import KNeighborsClassifier

ff=open("train.csv","r")
cc=csv.reader(ff)
info=[]
result=[]
csv_data = list(cc)

ll=len(csv_data)
for i in range(1,ll):
	#info.append([int(csv_data[i][0]),int(csv_data[i][2]),int(csv_data[i][5]),int(csv_data[i][6]),int(csv_data[i][7])])
	info.append([int(csv_data[i][2]),int(csv_data[i][6]),int(csv_data[i][7])])
	result.append(int(csv_data[i][1]))
#print info
#print result

#neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(info, result)

#print neigh.predict([[893,3,1,0],[894,2,0,0]])

ftest=open("test.csv","r")
test_data=list(csv.reader(ftest))
ltest=len(test_data)
test_info=[]
for j in range(1,ltest):
	test_info.append([int(test_data[j][1]),int(test_data[j][5]),int(test_data[j][6])])

#print test_info
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(info, result)
predict=neigh.predict(test_info)

print "PassengerId"+","+"Survived"
for k in range(892,1310):
	print str(k)+","+str(predict[k-892])

ff.close()
ftest.close()
