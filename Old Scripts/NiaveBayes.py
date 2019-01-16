import pandas as pd

#my homemade functions for auc
import rocfunctions as rc

#for niave bayes
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

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

	
# 	print("10 Fold average accuracy:",sum(acc)/10)

# 	nb.fit(factors,response)
# 	ypred = nb.predict(factors)

# 	acc = metrics.accuracy_score(response,ypred)

# 	print("Full fit internal accuracy:",acc)	
# 	return()

#for the titles.
# print("For the titles:")
# cross_NB(titles,titles_r)

#Average acc is .569
#full fit acc is .585

#for the summaries:
# print("For the summaries:")
# cross_NB(summaries,summaries_r)

#ave acc .648
#full fir is .680

##############################################################
#ROC analysis of NB

nb = GaussianNB()
nb.fit(titles,titles_r)

pp = nb.predict_proba(titles)
order = nb.classes_

iss_roc_t = rc.cat_ROC("iss", pp, order, titles_r)
som_roc_t = rc.cat_ROC("som", pp, order, titles_r)
not_roc_t = rc.cat_ROC("not", pp, order, titles_r)

# wa = rc.weighedAUC(iss_roc_t,som_roc_t,not_roc_t,titles_r)
# print("The weighed AUC for the titles:",wa)

#I am getting .710
#Now for the summaries: 

# nb.fit(summaries,summaries_r)

# pp = nb.predict_proba(summaries)
# order = nb.classes_

# sin = rc.cat_ROC("iss", pp, order, summaries_r)
# ssn = rc.cat_ROC("som", pp, order, summaries_r)
# not_roc_s = rc.cat_ROC("not", pp, order, summaries_r)

# wa = rc.weighedAUC(iss_roc_s,som_roc_s,not_roc_s,summaries_r)
# print("The weighed AUC for the summaries:",wa)

#I am getting .809

#########################################################
#lets plot:

import matplotlib.pyplot as plt

#Now lets try to plot it.
x0, y0 = zip(*iss_roc_t)
x1, y1 = zip(*som_roc_t)
x2, y2 = zip(*not_roc_t)

plt.plot(x0,y0,label="Controversal")
plt.plot(x1,y1,label="Somewhat Controversal")
plt.plot(x2,y2,label="Not Controversal")
plt.legend(loc="lower right")
plt.show()
