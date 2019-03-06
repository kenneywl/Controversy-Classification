import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000) 

# art = pd.read_pickle("articles_m.pkl")
art = pd.read_pickle('art_m.pkl')

# To get rid of the errors
no_error = []
for i in art.index:
	if not isinstance(art.loc[i,'P/N Metric'], str):
		no_error += [i]

art = art.loc[no_error]

#To whittle down to which titles are "real"
# ah = art['Title_Closeness'] > .01
# art = art[ah]

#separate in to cont and ncont
art_n = art[art.Response == 'not controversial']
art_c = art[art.Response == 'controversial']

nc = "Not Controversial"
cc = "Controversial"

##############################################################
#Just naive bayes

# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import KFold
# from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

# factors = art.loc[:,['Pos Score','Neg Score']]
# response = art.loc[:,'Response']

# nb = GaussianNB()
# kf = KFold(n_splits=10)
# kf.get_n_splits(factors)

# #for acc
# pred_response = np.ndarray(shape=(0,))
# #for auc
# pred_probability = np.ndarray(shape=(0,2))
# for i,j in kf.split(factors):
# 	nb.fit(factors.iloc[i,:],response.iloc[i])

# 	pv = nb.predict(factors.iloc[j,:]) #predict the outcome
# 	pp = nb.predict_proba(factors.iloc[j,:]) #predict the probabilities

# 	pred_response = np.append(pred_response,pv) #apppend to the master list
# 	pred_response = pd.DataFrame(pred_response)

# 	pred_probability = np.append(pred_probability,pp,axis=0) #append to the master list
# 	pred_probability = pd.DataFrame(pred_probability,columns=list(nb.classes_))


# pred_accuracy = accuracy_score(response,pred_response)
# print(pred_accuracy)

# pred_auc = roc_auc_score(response,pred_probability.iloc[:,1])
# print(pred_auc)

# fpr, tpr, _ = roc_curve(response,pred_probability.iloc[:,1], pos_label='not controversial')

# plt.plot(fpr,tpr)
# plt.show()

##############################################################################################
# f1 = plt.figure(1)
# plt.scatter(art_n['Neg Score'], art_n['Pos Score'],label=nc,alpha=.5)
# plt.scatter(art_c['Neg Score'], art_c['Pos Score'],label=cc,alpha=.5)
# plt.xlabel("Negative Score")
# plt.ylabel("Positive Score")
# plt.legend(loc="upper right")
# plt.title("Emotional Content")
# f1.show()

# # # # # #
# f2 = plt.figure(2)
# plt.hist(list(art_n['PN Metric']), label=nc,density=True,alpha=.5)
# plt.hist(list(art_c['PN Metric']), label=cc,density=True,alpha=.5)
# plt.title("Multiplication Metric")
# plt.xlabel("Positive Score * Negative Score")
# plt.ylabel("Frequency")
# plt.legend()
# f2.show()

f3 = plt.figure(3)
plt.hist(list(art_n['P/N Metric']),label=nc,density=True,alpha=.5)
plt.hist(list(art_c['P/N Metric']),label=cc,density=True,alpha=.5)
plt.title("Division Metric")
plt.xlabel("Positive Score / Negative Score")
plt.ylabel("Frequency")
plt.xlim(-1,15)
plt.legend()
f3.show()

# f6 = plt.figure(4)
# plt.hist(list(art_n['Word Count']),label=nc,density=True, alpha=.5)
# plt.hist(list(art_c['Word Count']),label=cc,density=True, alpha=.5)
# plt.title("Word Counts")
# plt.xlabel("Word Count")
# plt.ylabel("Frequency")
# plt.xlim(0,1300)
# plt.legend()
# f6.show()

#######################################
# f4 = plt.figure(5)
# plt.hist(list(art_n['P+N Metric']), label=nc,density=True,alpha=.5)
# plt.hist(list(art_c['P+N Metric']), label=cc,density=True,alpha=.5)
# plt.title("Addition Metric")
# plt.legend()
# f4.show()

# f5 = plt.figure(6)
# plt.hist(list(art_n['P-N Metric']), label=nc,density=True,alpha=.5)
# plt.hist(list(art_c['P-N Metric']), label=cc,density=True,alpha=.5)
# plt.title("Subtraction Metric")
# plt.legend()
# f5.show()