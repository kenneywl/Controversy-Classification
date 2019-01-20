from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score,roc_curve,auc,accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

##########################################################################################
##########################################################################################

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


def plot_auc(factors,response,k):
	pp = prob_cross(factors,response,k)
	bin_response = label_binarize(response, classes=list(pp))

	tpr = dict()
	fpr = dict()
	roc_auc = dict()
	for i in range(3):
		fpr[i], tpr[i], _ = roc_curve(bin_response[:,i],pp.iloc[:,i])

	interp_tpr = [tpr[0], np.interp(fpr[0],fpr[1],tpr[1]), np.interp(fpr[0],fpr[2],tpr[2])]

	value = [response.value_counts()["Controversial"], response.value_counts()["Somewhat Controversial"], response.value_counts()["Not Controversial"]]

	vert_average = []
	for i in range(len(fpr[0])):
		vert = 0
		for j in range(3):
			vert += interp_tpr[j][i]*value[j]
		vert_average += [vert]

	vert_average = [vert_average[i]/sum(value) for i in range(len(fpr[0]))]

	plot_title = str(k) + " Nearest Neighbors " + response.name + " Binary Relevance ROC"

	f1 = plt.figure()
	plt.plot(fpr[0],tpr[0],label="Controversial")
	plt.plot(fpr[1],tpr[1],label="Somewhat Controversial")
	plt.plot(fpr[2],tpr[2],label="Not Controversial")
	plt.plot(fpr[0],vert_average,label="Vertical Weighed Average")
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.title(plot_title)
	f1.show()

#Plot acc and max AUC by k

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

def which_k_plot(factors,response):
	max_k = factors.shape[1]
	weighed_auc = []
	acc = []
	for i in range(1,max_k+1):
		pp_response = prob_cross(factors,response,i)
		bin_response = label_binarize(response, classes=list(pp_response))

		weighed_auc += [roc_auc_score(bin_response,pp_response,average="weighted")]
		acc += [acc_cross(factors,response,i)]

	plot_title = response.name + " Weighed AUC and Accuracy by k"

	k_list = [i+1 for i in range(max_k)]
	f1 = plt.figure()
	plt.plot(k_list,weighed_auc,label="Weighed AUC")
	plt.plot(k_list,acc,label="Accuracy")
	plt.xlabel('K Values')
	plt.ylabel('Probability')
	plt.legend(loc="lower right")
	plt.title(plot_title)
	f1.show()