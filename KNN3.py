import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold

#homebrew for plotting.

from KNN_plot_func import *

#Import our data:
summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

#This gets accuracy and weighed AUC for all values of K. This requires acc_cross and prob_cross below.
#acc_cross gets the 10 fold cross validation, and prob_cross get the 10 fold cross probabilites
#for each category. It outputs a tuple, (weighed AUC, Accuracy)

def which_k(factors,response):
	max_k = factors.shape[1]
	weighed_auc = []
	acc = []
	for i in range(1,max_k+1):
		pp_response = prob_cross(factors,response,i)
		bin_response = label_binarize(response, classes=list(pp_response))

		weighed_auc += [roc_auc_score(bin_response,pp_response,average="weighted")]
		acc += [acc_cross(factors,response,i)]

	auc_max = max(weighed_auc)
	acc_max = max(acc)
	k_auc_max = weighed_auc.index(auc_max)+1
	k_acc_max = acc.index(acc_max)+1

	return(("AUC max info:", k_auc_max, auc_max),("Accuracy max info:", k_acc_max, acc_max))

def acc_cross(factors,response,k):
	kn = KNeighborsClassifier(n_neighbors=k)
	kf = KFold(n_splits=10)
	kf.get_n_splits(factors)

	acc = []
	for i,j in kf.split(factors):
		kn.fit(factors.iloc[i,:],response.iloc[i])
		pred = kn.predict(factors.iloc[j,:])
		acc += [accuracy_score(response.iloc[j],pred)]
	acc_m = sum(acc)/10
	return(acc_m)

def prob_cross(factors,response,k):
	kf = KFold(n_splits=10)
	kf.get_n_splits(factors)

	kn = KNeighborsClassifier(n_neighbors=k)
	
	pp = np.ndarray(shape=(0,3))
	for i,j in kf.split(factors):
		kn.fit(factors.iloc[i,:],response.iloc[i])
		pp = np.append(pp,kn.predict_proba(factors.iloc[j,:]),axis=0)

	pp = pd.DataFrame(pp,columns=list(kn.classes_))
	return pp

#########################################################################################
#Let us the above functions to do this.

print("For the titles:")
ans_t = which_k(titles,titles_r)
print(ans_t)

print("For the summaries:")
ans_s = which_k(summaries,summaries_r)
print(ans_s)


#Output looks like this:

# For the titles:
# ('AUC max info:', 10, 0.6036526659216251) 
#  ('Accuracy max info:', 32, 0.49505617977528094)
# For the summaries:
# ('AUC max info:', 14, 0.6857727840339154) 
#  ('Accuracy max info:', 12, 0.65)

#lets plot it all:

plot_auc(titles,titles_r,10)
plot_auc(summaries,summaries_r,14)

which_k_plot(titles,titles_r)
which_k_plot(summaries,summaries_r)