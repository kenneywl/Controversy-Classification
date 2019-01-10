import pandas as pd

#my homemade functions for auc
import rocfunctions as rc

#for niave bayes
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

#for plotting
import matplotlib.pyplot as plt


summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

# def cross_NB(factors,response):
# 	hold = int(factors.shape[0]/10)

# 	nb = GaussianNB()
# 	acc = []
# 	for i in range(10):
# 		ind = [j for j in range(hold*i,hold*(i+1))]

# 		set_p = factors.drop(ind)
# 		set_r = response.drop(ind)

# 		nb.fit(set_p,set_r)
# 		ypred = nb.predict(factors.iloc[ind,:])

# 		acc += [metrics.accuracy_score(response.iloc[ind],ypred)]

# 	print("10 Fold, Average Accuracy:",sum(acc)/10)

# 	return()

#for the titles.
# print("For the titles:")
# cross_NB(titles,titles_r)

#Average acc is .569

#for the summaries:
# print("For the summaries:")
# cross_NB(summaries,summaries_r)

#.648 accuracy.

##############################################################
#ROC analysis of NB

nb = GaussianNB()
nb.fit(titles,titles_r)

pp = nb.predict_proba(titles)

iss_roc = rc.cat_ROC("iss", pp, titles_r)
som_roc = rc.cat_ROC("som", pp, titles_r)
not_roc = rc.cat_ROC("not", pp, titles_r)

wa = rc.weighedAUC(iss_roc,som_roc,not_roc,titles_r)
print("The weighed AUC for the titles:",wa)

#I am getting .487
#Now for the summaries: 

nb.fit(summaries,summaries_r)

pp = nb.predict_proba(summaries)

iss_roc = rc.cat_ROC("iss", pp, summaries_r)
som_roc = rc.cat_ROC("som", pp, summaries_r)
not_roc = rc.cat_ROC("not", pp, summaries_r)

wa = rc.weighedAUC(iss_roc,som_roc,not_roc,summaries_r)
print("The weighed AUC for the summaries:",wa)

#I am getting .620

######################
#for plotting.

x, y = zip(*not_roc)

plt.scatter(x,y)
plt.show()