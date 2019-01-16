from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score,roc_curve,auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

bin_resp_titles = label_binarize(titles_r, \
	classes=["Controversial","Not Controversial","Somewhat Controversial"])

nb = GaussianNB()
nb.fit(titles,titles_r)

pp_titles = nb.predict_proba(titles)
weighed_auc = roc_auc_score(bin_resp_titles,pp_titles,average=None)

print(weighed_auc)
