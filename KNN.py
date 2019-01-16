import pandas as pd
from sklearn.model_selection import train_test_split

#For KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#####################################################################
#we unpickle our data from cleandata.py

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

######################################################################
#I wrote a function to do a 10 fold cross validation.
#It leaves out 10% at a time and tests against that.
#This function finds the best value for k


def knn_which_k(factors,response,max_k=31):
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

	max_k = acc_m.index(max(acc_m))
	print("max k:",max_k+1,"\nmax acc:",max(acc_m))

	return()

################################################################
#The accuracies for each value of k from 1 to 30
#for titles:

print("For the titles:")
knn_which_k(titles,titles_r)

#The k value is 22 with an acc of .496
#somewhat better than random (which would be .33)
#as there are three possible response categaores.

#For the summaries:
print("For the summaries:")
knn_which_k(summaries,summaries_r)

#The k value is 11, with a acc of .652
#not terrible.
