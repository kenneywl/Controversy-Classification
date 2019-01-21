import pandas as pd
from math import log2
from sklearn.metrics import roc_auc_score, roc_curve,auc

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

fa = pd.read_excel("Factors.xlsx")
ti = pd.read_excel("Titles.xlsx")
su = pd.read_excel("Summaries.xlsx")

titles_a = pd.DataFrame(ti.iloc[:,[2,3,4]])
summaries_a = pd.DataFrame(su.iloc[:1000,[3,4,5]])

summaries_a.iloc[:,2] = [float(summaries_a.iloc[i,2]) for i in range(1000)]

summaries_a = summaries_a.rename( \
	columns={"controversial":"Controversial","not controversial":"Not Controversial","somewhat controversial":"Somewhat Controversial"})

margins_t = titles_a.sum(numeric_only=True).div(titles_a.shape[0]*20)
margins_s = summaries_a.sum(numeric_only=True).div(summaries_a.shape[0]*20)

con_t = [i/2 for i in titles_a["Somewhat Controversial"]]
con_s = [i/2 for i in summaries_a["Somewhat Controversial"]]

titles_a["Controversial"] = [i+j for i,j in zip(con_t,titles_a["Controversial"])]
titles_a["Not Controversial"] = [i+j for i,j in zip(con_t,titles_a["Not Controversial"])]

summaries_a["Not Controversial"] = [i+j for i,j in zip(con_s,summaries_a["Not Controversial"])]
summaries_a["Controversial"] = [i+j for i,j in zip(con_s,summaries_a["Controversial"])]

titles_a = titles_a.drop(labels="Somewhat Controversial",axis=1)
summaries_s = summaries_a.drop(labels="Somewhat Controversial",axis=1)

#####################################################################################################
#####################################################################################################

#We choose max to be the response, and get an entropy base two for each (to possibly weigh each instance)
#how would you do this?
#Ties go to not controversial.
titles_res = []
for i in range(1000):
	if titles_a["Controversial"][i] > titles_a["Not Controversial"][i]:
		titles_res += ["Controversial"]
	else:
		titles_res += ["Not Controversial"]

# titles_ent = []
# for i in range(1000):
#  	titles_ent += [log2(20)-(titles_a.iloc[i,0]*log2(titles_a.iloc[i,0])+titles_a.iloc[i,1]*log2(titles_a.iloc[i,1]))/20]

# titles_ent = [1-i for i in titles_ent]

####################################################################################################
#Yet another approch is have the response a value between 0 and 1.
#the response is then p(Controversial)

# titles_pzero = []
# for i in range(1000):
# 	titles_pzero += [titles_a["Controversial"][i]/20]

# print(titles_zero)

######################################################################################################
######################################################################################################
#ties go to not controversial
summaries_res = []
for i in range(1000):
	if summaries_a["Controversial"][i] > summaries_a["Not Controversial"][i]:
		summaries_res += ["Controversial"]
	else:
		summaries_res += ["Not Controversial"]
######

count = 0
for i,j in zip(summaries_res,titles_res):
	if i!=j:
		count += 1

count2 = 0
for i,j in zip(summaries_r,titles_r):
	if i != j:
		count2 += 1


print(count/len(summaries_res))
print(count2/len(summaries_r))

from NaiveBayes3 import *

# titles_res = pd.Series(titles_res)
# which_k(titles,titles_res)

summaries_res = pd.Series(summaries_res)

print(summaries_res.shape)
pp = NB_prob_cross(summaries,summaries_res)

print(pp.shape)
bin_response = label_binarize(summaries_res.iloc[:940],classes=list(pp))
weighed_auc = roc_auc_score(bin_response[:940],pp.iloc[:,1])
print(weighed_auc)