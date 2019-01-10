import pandas as pd
import rocfunctions as rc

#For KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#####################################################################
#we can unpickle our data from cleandata.py

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

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

	max_k = acc_m.index(max(acc_m))
	print("max k:",max_k+1,"\nmax acc:",max(acc_m))

	return()

################################################################
#The accuracies for each value of k from 1 to 30
#for titles:

# print("For the titles:")
# knn_cross(titles,titles_r)

#The k value is 22 with a acc of .496
#somewhat better than random (which would be .33)
#as there are three possible response categaores.

#For the summaries:
# print("For the summaries:")
# knn_cross(summaries,summaries_r)

#The k value is 11, with a acc of .652
#not terrible.

##################################################################
##################################################################
#Now lets explore the ROC curve. We use the max k values above.

knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(summaries,summaries_r)

pp = knn.predict_proba(summaries)

iss_roc = rc.cat_ROC("iss", pp, summaries_r)
som_roc = rc.cat_ROC("som", pp, summaries_r)
not_roc = rc.cat_ROC("not", pp, summaries_r)

#The above are ROC data points.
#now we have three roc curve data points. to compute the discrete integral
#of eachand weigh according to class probabilities and sum up.

wa = rc.weighedAUC(iss_roc,som_roc,not_roc,summaries_r)
print("The weighed AUC for the summaries:",wa)

#weighed AUC for each category is: .625

###################################################################################
#Now lets do the same with titles.

# knn = KNeighborsClassifier(n_neighbors = 22)
# knn.fit(titles,titles_r)

# pp = knn.predict_proba(titles)

# iss_roc = rc.cat_ROC("iss", pp, titles_r)
# som_roc = rc.cat_ROC("som", pp, titles_r)
# not_roc = rc.cat_ROC("not", pp, titles_r)

# wa = rc.weighedAUC(iss_roc,som_roc,not_roc,titles_r)
# print("The weighed AUC for the titles:",wa)

#weighed AUC is .486