import pandas as pd

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

# def knn_cross(factors,response,max_k=31):
# 	hold = int(factors.shape[0]/10)

# 	acc_m = []
# 	for k in range(1,max_k):
# 		knn = KNeighborsClassifier(n_neighbors = k)
# 		acc = []
# 		for i in range(10):
# 			ind = [j for j in range(hold*i,hold*(i+1))]

# 			set_p = factors.drop(ind)
# 			set_r = response.drop(ind)

# 			knn.fit(set_p,set_r)
# 			ypred = knn.predict(factors.iloc[ind,:])

# 			acc += [metrics.accuracy_score(response.iloc[ind],ypred)]
# 		acc_m += [sum(acc)/10]

# 	max_k = acc_m.index(max(acc_m))
# 	print("max k:",max_k+1,"\nmax acc:",max(acc_m))

# 	return()

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
#The below function takes class probabilites and actual probs
#and the category you are interested in and makes the roc data points.

def cat_ROC(cat, pred_prob, actual_cat):

	acc_m = []
	for j in range(-1,102):
		iss_li = []
		for i in range(pp.shape[0]):
			if pred_prob[i,1] >= j/100:
				iss_li += [cat]
			else:
				iss_li += ["not " + cat]

		truep_rate = 0
		falsep_rate = 0
		for i in range(pred_prob.shape[0]):
			if iss_li[i] == actual_cat.iloc[i]:
				truep_rate += 1

			not_iss = iss_li[i] == cat
			not_iss2 = actual_cat[i] != cat
			if not_iss and not_iss2:
				falsep_rate += 1

		cat_list = list(actual_cat.value_counts().index)
		cat_ind = cat_list.index(cat)

		count = actual_cat.value_counts()

		truep_rate /= count[cat_ind]
		falsep_rate /= sum(count) - count[cat_ind]

		acc_m += [(falsep_rate,truep_rate)]

	acc_m = list(set(acc_m))
	acc_m = sorted(acc_m,key=lambda x: x[0])
	acc_m = sorted(acc_m,key=lambda x: x[1])

	return(acc_m)

####################################################################################
####################################################################################
knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(summaries,summaries_r)

pp = knn.predict_proba(summaries)


iss_roc = cat_ROC("iss", pp, summaries_r)
som_roc = cat_ROC("som", pp, summaries_r)
not_roc = cat_ROC("not", pp, summaries_r)

###################################################################################
#now we have three roc curve data points. to compute the discrete integral
#of eachand weigh according to class probabilities and sum up.

from scipy.integrate import cumtrapz

def cumsum(data):
	x = [i[0] for i in data]
	y = [i[1] for i in data]

	auc_v = list(cumtrapz(y,x))
	return(auc_v)

aucs = [max(cumsum(not_roc)),max(cumsum(som_roc)),max(cumsum(iss_roc))]

aucs_weighed = [aucs[i]*summaries_r.value_counts()[i]/summaries_r.shape[0] for i in range(3)]
print(sum(aucs_weighed))

#weighed AUC for each category is: .625

###################################################################################
#Now lets do the same with titles.

# knn = KNeighborsClassifier(n_neighbors = 22)
# knn.fit(titles,titles_r)

# pp = knn.predict_proba(titles)


# iss_roc = cat_ROC("iss", pp, titles_r)
# som_roc = cat_ROC("som", pp, titles_r)
# not_roc = cat_ROC("not", pp, titles_r)

# from scipy.integrate import cumtrapz

# def cumsum(data):
# 	x = [i[0] for i in data]
# 	y = [i[1] for i in data]

# 	auc_v = list(cumtrapz(y,x))
# 	return(auc_v)

# aucs = [max(cumsum(not_roc)),max(cumsum(som_roc)),max(cumsum(iss_roc))]

# aucs_weighed = [aucs[i]*titles_r.value_counts()[i]/titles_r.shape[0] for i in range(3)]
# print(sum(aucs_weighed))

#weighed AUC is .591