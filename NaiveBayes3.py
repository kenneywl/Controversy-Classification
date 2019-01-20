import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize

#Lets import our data:
summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

##########################################################################

def NB_prob_cross(factors,response):
	model = GaussianNB()
	kf = KFold(n_splits=10)
	kf.get_n_splits(factors)

	pp = np.ndarray(shape=(0,3))
	for i,j in kf.split(factors):
		model.fit(factors.iloc[i,:],response.iloc[i])
		pred = model.predict_proba(factors.iloc[j,:])
		pp = np.append(pp,pred,axis=0)
	pp = pd.DataFrame(pp,columns=list(model.classes_))
	return(pp)

def NB_plot_auc(factors,response):
	pp = NB_prob_cross(factors,response)
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

	plot_title = response.name + " Niave Bayes Binary Relevance ROC"

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

def NB_print_AUC(factors,response):
	pp = NB_prob_cross(factors,response)
	bin_response = label_binarize(response, classes=list(pp))
	weighed_auc = roc_auc_score(bin_response,pp,average="weighted")

	message = " AUC:"
	print(message,weighed_auc)