import cleandata as cd

#cleandata holds all the cleaning and positioning for analysis.
#The following has info:
#cd.info()

#For KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
######################################################################
#I wrote a function to do a 10 fold cross validation.
#this function requires the "for KNN above."

def knn_cross(factors,response,max_k=31):
	hold = int(factors.shape[0]/10)

	acc_m = []
	for k in range(1,max_k):
		knn = KNeighborsClassifier(n_neighbors = k)
		acc = []
		for i in range(10):
			ind = [j for j in range(hold*i,hold*(i+1))]

			set_p = factors.drop(ind)
			set_r = response.drop(ind)

			knn.fit(set_p,set_r)
			ypred = knn.predict(factors.iloc[ind,:])

			acc += [metrics.accuracy_score(response.iloc[ind],ypred)]
		acc_m += [sum(acc)/10]
	return(acc_m)

################################################################
#The accuracies for each value of k from 1 to 30

#For the summaries:
# acc = knn_cross(cd.summaries,cd.summaries_r)
# [print(i+1,acc[i]) for i in range(len(acc))]
# max_k = acc.index(max(acc))
# print("max k:",max_k+1,"\nmax acc:",max(acc))

#The k value is 11, with an average acc of .652
#not terrible.

#for titles:
acc = knn_cross(cd.titles,cd.titles_r)
[print(i+1,acc[i]) for i in range(len(acc))]
max_k = acc.index(max(acc))
print("max k:",max_k+1,"\nmax acc:",max(acc))

#The k value is 22 with an average value of .496
#somewhat better than random (which would be .33)
#as there are three possible response categaores.