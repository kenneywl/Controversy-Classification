from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score,roc_curve,auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

#The following is to get the probabilities of each
#category by ten fold cross.

def knn_cross(factors,response,k):
	hold = int(factors.shape[0]/10)
	kn = KNeighborsClassifier(n_neighbors=k)
	pp = np.ndarray(shape=(0,3))
	for i in range(9):
		ind = [j for j in range(hold*i,hold*(i+1))]

		set_p = factors.drop(ind)
		set_r = response.drop(ind)

		kn.fit(set_p,set_r)
		pp = np.append(pp,kn.predict_proba(factors.iloc[ind,:]),axis=0)


	ind = [j for j in range(hold*9,len(response))]

	set_p = factors.drop(ind)
	set_r = response.drop(ind)

	kn.fit(set_p,set_r)
	pp = np.append(pp,kn.predict_proba(factors.iloc[ind,:]),axis=0)
	return pp

##########################################################################################
##########################################################################################
#Now lets get probs with our function above
#I used 21 as it was the max value from KNN.py
pp_titles = knn_cross(titles,titles_r,10)

bin_resp_titles = label_binarize(titles_r, \
	classes=["Controversial","Not Controversial","Somewhat Controversial"])

weighed_auc = roc_auc_score(bin_resp_titles,pp_titles,average="weighted")

print("10 Nearest Neighbors Titles AUC weighed by class probabilites:",weighed_auc)

tpr = dict()
fpr = dict()
roc_auc = dict()
for i in range(3):
	fpr[i], tpr[i], _ = roc_curve(bin_resp_titles[:,i],pp_titles[:,i])

#now to get the averaged curve I need to interp

interp_tpr = [tpr[0], np.interp(fpr[0],fpr[1],tpr[1]), np.interp(fpr[0],fpr[2],tpr[2])]

#in this order "som" "iss" "not"
value = [titles_r.value_counts()["Controversial"], titles_r.value_counts()["Somewhat Controversial"], \
         titles_r.value_counts()["Not Controversial"]]

vert_average = []
for i in range(len(fpr[0])):
	vert = 0
	for j in range(3):
		vert += interp_tpr[j][i]*value[j]
	vert_average += [vert]

vert_average = [vert_average[i]/sum(value) for i in range(len(fpr[0]))]

f1 = plt.figure()
plt.plot(fpr[0],tpr[0],label="Controversial")
plt.plot(fpr[1],tpr[1],label="Somewhat Controversial")
plt.plot(fpr[2],tpr[2],label="Not Controversial")
plt.plot(fpr[0],vert_average,label="Vertical Weighed Average")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title("10 Nearest Neighbors Titles Binary Relevance ROC")
f1.show()

#Titles
#k:10 AUC:.59877

###################################################################################
#Now lets get probs with our function above
#I used 11 as it was the max value for summaries from KNN.py

pp_summaries = knn_cross(summaries,summaries_r,14)

bin_resp_summaries = label_binarize(summaries_r, \
	classes=["Controversial","Not Controversial","Somewhat Controversial"])

weighed_auc = roc_auc_score(bin_resp_summaries,pp_summaries,average="weighted")

print("14 Nearest Neighbors Summaries AUC weighed by class probabilites:",weighed_auc)

tpr = dict()
fpr = dict()
roc_auc = dict()
for i in range(3):
	fpr[i], tpr[i], _ = roc_curve(bin_resp_summaries[:,i],pp_summaries[:,i])

#now to get the averaged curve I need to interp

interp_tpr = [tpr[0], np.interp(fpr[0],fpr[1],tpr[1]), np.interp(fpr[0],fpr[2],tpr[2])]

#in this order "som" "iss" "not"
value = [summaries_r.value_counts()["Controversial"], summaries_r.value_counts()["Somewhat Controversial"], \
         summaries_r.value_counts()["Not Controversial"]]

vert_average = []
for i in range(len(fpr[0])):
	vert = 0
	for j in range(3):
		vert += interp_tpr[j][i]*value[j]
	vert_average += [vert]

vert_average = [vert_average[i]/sum(value) for i in range(len(fpr[0]))]

f1 = plt.figure()
plt.plot(fpr[0],tpr[0],label="Controversial")
plt.plot(fpr[1],tpr[1],label="Somewhat Controversial")
plt.plot(fpr[2],tpr[2],label="Not Controversial")
plt.plot(fpr[0],vert_average,label="Vertical Weighed Average")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title("14 Nearest Neighbors Summaries Binary Relevance ROC")
f1.show()


#Titles
#k:10 AUC:.59877

#Summaries
#k;14 AUC:.68577