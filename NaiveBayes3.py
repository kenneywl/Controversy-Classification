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

def nb_cross(factors,response,model):
	kf = KFold(n_splits=10)
	kf.get_n_splits(factors)

	pp = np.ndarray(shape=(0,3))
	for i,j in kf.split(factors):
		model.fit(factors.iloc[i,:],response.iloc[i])
		pred = model.predict_proba(factors.iloc[j,:])
		pp = np.append(pp,pred,axis=0)

	return(pp)


####################################################################################
#For titles:
model = GaussianNB()
pp_titles = nb_cross(titles,titles_r,model)

bin_resp_titles = label_binarize(titles_r, \
	classes=["Controversial","Not Controversial","Somewhat Controversial"])

weighed_auc = roc_auc_score(bin_resp_titles,pp_titles,average="weighted")

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

f = plt.figure()
plt.plot(fpr[0],tpr[0],label="Controversial")
plt.plot(fpr[1],tpr[1],label="Somewhat Controversial")
plt.plot(fpr[2],tpr[2],label="Not Controversial")
plt.plot(fpr[0],vert_average,label="Vertical Weighed Average")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title("Naive Bayes Titles Binary Relevance ROC")
f.show()
print("Naive Bayes Titles AUC weighed by class probabilites:",weighed_auc)
#.68067

####################################################################################################

model = GaussianNB()
pp_summaries = nb_cross(summaries,summaries_r,model)

bin_resp_summaries = label_binarize(summaries_r, \
	classes=["Controversial","Not Controversial","Somewhat Controversial"])

weighed_auc = roc_auc_score(bin_resp_summaries,pp_summaries,average="weighted")

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
plt.title("Naive Bayes Summaries Binary Relevance ROC")
f1.show()
print("Naive Bayes Summaries AUC weighed by class probabilites:",weighed_auc)
#.8155