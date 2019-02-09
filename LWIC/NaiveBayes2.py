import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

########################################################################################
#lets make a function to do 10 fold cross for niave bayes.
#This is very much like KNN but a little different.

def nb_auc_acc_cross(factors,response):
	nb = GaussianNB()
	kf = KFold(n_splits=10)
	kf.get_n_splits(factors)

	#for acc
	pred_response = np.ndarray(shape=(0,))
	#for auc
	pred_probability = np.ndarray(shape=(0,3))
	for i,j in kf.split(factors):
		nb.fit(factors.iloc[i,:],response.iloc[i])

		pv = nb.predict(factors.iloc[j,:]) #predict the outcome
		pp = nb.predict_proba(factors.iloc[j,:]) #predict the probabilities

		pred_response = np.append(pred_response,pv) #apppend to the master list
		pred_response = pd.DataFrame(pred_response)

		pred_probability = np.append(pred_probability,pp,axis=0) #append to the master list
		pred_probability = pd.DataFrame(pred_probability,columns=list(nb.classes_))
	return(pred_probability,pred_response)

#####################################################################################


def nb_print_plot(factors,response):
	#Lets do this first for titles
	#get the 10 fold cross info.

	pred_probability, pred_response = nb_auc_acc_cross(factors,response)

	#for acc
	pred_accuracy = accuracy_score(response,pred_response)

	#for auc
	bin_response = label_binarize(response, classes=list(pred_probability))
	pred_auc = roc_auc_score(bin_response,pred_probability,average="weighted")
	print(" AUC:",pred_auc,"\n ACC:",pred_accuracy)

	#now we have preditcted acc and predicted auc (and printed it.)
	#now lets get graphs:

	#this get us the true positive rate and the false positive rate.
	tpr = dict()
	fpr = dict()
	roc_auc = dict()
	for i in range(3):
		fpr[i], tpr[i], _ = roc_curve(bin_response[:,i],pred_probability.iloc[:,i])

	#now to get the averaged curve I need to interpetation:

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

	plot_title = response.name + " Niave Bayes Binary Relevance ROC"

	f = plt.figure()
	plt.plot(fpr[0],tpr[0],label="Controversial")
	plt.plot(fpr[1],tpr[1],label="Somewhat Controversial")
	plt.plot(fpr[2],tpr[2],label="Not Controversial")
	plt.plot(fpr[0],vert_average,label="Vertical Weighed Average")
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.title(plot_title)
	f.show()