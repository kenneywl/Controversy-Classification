import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import numpy as np

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

#This function computes the probabilities 10% at a time to get the whole
#data set by training on the other 90%. If the data set has an uneven amount
#of instances, the 10% training set is rounded down to the nearest whole number
#and the last 10% has the extra instances. This means that we train on slightly
#over 90% and train on slightly less than 10%, except for the last 10%
#which has this reversed.


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
	pp = pd.DataFrame(pp,columns=kn.classes_)
	return pp


def which_k_auc(factors,response):

	max_k = factors.shape[1]
	weighed_auc = []
	for i in range(1,max_k+1):
		pp_response = knn_cross(factors,response,i)
		bin_response = label_binarize(response, \
						classes=list(pp_response))

		weighed_auc += [roc_auc_score(bin_response,pp_response,average="weighted")]
	return(weighed_auc)


ans_t = which_k_auc(titles,titles_r)
ans_t_max = ans_t.index(max(ans_t))+1

ans_s = which_k_auc(summaries,summaries_r)
ans_s_max = ans_s.index(max(ans_s))+1


print("For the titles:")
print("k for max weighed AUC:",ans_t_max)
print("AUC for max k:", max(ans_t))

#k: 10 AUC value: .59877

print("For the summaries:")
print("k for max weighed AUC:",ans_s_max)
print("AUC for max k:", max(ans_s))

#k: 14 AUC value: .68577