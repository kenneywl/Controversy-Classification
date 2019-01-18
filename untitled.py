from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score,roc_curve,auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

def nb_cross(factors,response):
	hold = int(factors.shape[0]/10)
	nb = GaussianNB()
	pp = np.ndarray(shape=(0,3))
	for i in range(9):
		ind = [j for j in range(hold*i,hold*(i+1))]

		set_p = factors.drop(ind)
		set_r = response.drop(ind)

		nb.fit(set_p,set_r)
		pp = np.append(pp,nb.predict_proba(factors.iloc[ind,:]),axis=0)

	ind = [j for j in range(hold*9,len(response))]

	set_p = factors.drop(ind)
	set_r = response.drop(ind)

	nb.fit(set_p,set_r)
	pp = np.append(pp,nb.predict_proba(factors.iloc[ind,:]),axis=0)

	return pp

bin_resp_titles = label_binarize(titles_r, \
	classes=["Controversial","Not Controversial","Somewhat Controversial"])

pp_titles = nb_cross(titles,titles_r)

weighed_auc = roc_auc_score(bin_resp_titles,pp_titles,average="weighted")

print("Naive Bayes Titles AUC weighed by class probabilites:",weighed_auc)
