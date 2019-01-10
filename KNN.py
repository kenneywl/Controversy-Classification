import cleandata as cd
import pandas as pd

#cleandata holds all the cleaning and positioning for analysis.
#The following has info:
#cd.info()

#For KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
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
# knn_cross(cd.titles,cd.titles_r)

#The k value is 22 with a acc of .499
#somewhat better than random (which would be .33)
#as there are three possible response categaores.

#For the summaries:
# print("For the summaries:")
# knn_cross(cd.summaries,cd.summaries_r)

#The k value is 11, with a acc of .652
#not terrible.

##################################################################
##################################################################
#Now lets explore the ROC curve. We use the max k values above.


knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(cd.summaries,cd.summaries_r)

pp = knn.predict_proba(cd.summaries)
#These are in the order, "not","iss","som"

#lets start with "iss"

# acc_m = []
# for j in range(-1,102):
# 	iss_li = []
# 	for i in range(pp.shape[0]):
# 		if pp[i,1] >= j/100:
# 			iss_li += ["iss"]
# 		else:
# 			iss_li += ["not iss"]

# 	truep_rate = 0
# 	falsep_rate = 0
# 	for i in range(pp.shape[0]):
# 		if iss_li[i] == cd.summaries_r.iloc[i]:
# 			truep_rate += 1

# 		not_iss = iss_li[i] == "not iss"
# 		not_iss2 = (cd.summaries_r.iloc[i] == "not") or (cd.summaries_r.iloc[i] == "som")
# 		if not_iss and not_iss2:
# 			falsep_rate += 1

# 	count = cd.summaries_r.value_counts()

# 	truep_rate /= count[2]
# 	falsep_rate /= (count[0]+count[1])

#	acc_m += [(falsep_rate,truep_rate)]

# iss_roc_data = acc_m

#Now lets do "not"
#count in this order "not", "som", "iss"

# acc_m = []
# for j in range(-1,102):
# 	iss_li = []
# 	for i in range(pp.shape[0]):
# 		if pp[i,0] >= j/100:
# 			iss_li += ["not"]
# 		else:
# 			iss_li += ["not not"]

# 	truep_rate = 0
# 	falsep_rate = 0
# 	for i in range(pp.shape[0]):
# 		if iss_li[i] == cd.summaries_r.iloc[i]:
# 			truep_rate += 1

# 		not_iss = iss_li[i] == "not not"
# 		not_iss2 = (cd.summaries_r.iloc[i] == "iss") or (cd.summaries_r.iloc[i] == "som")
# 		if not_iss and not_iss2:
# 			falsep_rate += 1

# 	count = cd.summaries_r.value_counts()

# 	truep_rate /= count[0]
# 	falsep_rate /= (count[1]+count[2])

#	acc_m += [(falsep_rate,truep_rate)]

# not_roc_data = acc_m

#Now lets do "som"

acc_m = []
for j in range(-1,102):
	iss_li = []
	for i in range(pp.shape[0]):
		if pp[i,0] <= j/100:
			iss_li += ["som"]
		else:
			iss_li += ["not som"]

	truep_rate = 0
	falsep_rate = 0
	for i in range(pp.shape[0]):
		if iss_li[i] == cd.summaries_r.iloc[i]:
			truep_rate += 1

		not_iss = iss_li[i] == "som"
		not_iss2 = cd.summaries_r.iloc[i] != "som"
		if not_iss and not_iss2:
			falsep_rate += 1

	count = cd.summaries_r.value_counts()

	truep_rate /= count[1]
	falsep_rate /= (count[0]+count[2])

	acc_m += [(falsep_rate,truep_rate)]

som_roc_data = acc_m

###################################################################################
#now we have three roc curve data. to compute the AUC and weigh according
#to class probabilities we compute the discrete integral of each.

#som_roc_data
#not_roc_data
#iss_roc_data

# print(som_roc_data)
som_roc_data = list(set(som_roc_data))
som_roc_data = sorted(som_roc_data,key=lambda x: x[0])

print(som_roc_data)