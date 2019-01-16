import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")


#This function collapses all but the "reference" category.
#probabilities:

def collapse_proba(cat,probs,order):
	order = list(order)
	cat_ind = order.index(cat)

	new_probs = [probs[i][cat_ind] for i in range(len(probs))]
	return(new_probs)

#This collapses the responses into the reference cat or not.

def collapse_response(cat,labs):
	labs = list(labs)
	new_labs = []
	for i in labs:
		if i == cat:
			new_labs += [1]
		else:
			new_labs += [0]

	return(new_labs)


nb = GaussianNB()

#Lets fit the whole title dataset
nb.fit(titles,titles_r)

#prob of each category and the order
pp_titles = nb.predict_proba(titles)
order_titles = nb.classes_

#call the functions.
titles_prob_iss = collapse_proba("iss",pp_titles,order_titles)
titles_resp_iss = collapse_response("iss",titles_r)

titles_prob_som = collapse_proba("som",pp_titles,order_titles)
titles_resp_som = collapse_response("som",titles_r)

titles_prob_not = collapse_proba("not",pp_titles,order_titles)
titles_resp_not = collapse_response("not",titles_r)

#get false proba and true probs.
fpr_iss, tpr_iss, _ = roc_curve(titles_resp_iss,titles_prob_iss)
fpr_som, tpr_som, _ = roc_curve(titles_resp_som,titles_prob_som)
fpr_not, tpr_not, _ = roc_curve(titles_resp_not,titles_prob_not)

#set up the plot.

f1 = plt.figure(1)
plt.plot(fpr_iss,tpr_iss,label="Controversial")
plt.plot(fpr_som,tpr_som,label="Somewhat Controversial")
plt.plot(fpr_not,tpr_not,label="Not Controversial")

#and lets go ahead and plot it:
plt.legend(loc="lower right")
plt.title("Naive Bayes Title Binary Relevance ROC")
#f1.show()

#And lets get an AUC score. We weigh each individual AUC by
#class probabilies.

bin_auc = [roc_auc_score(titles_resp_som,titles_prob_som), \
          + roc_auc_score(titles_resp_iss,titles_prob_iss), \
          + roc_auc_score(titles_resp_not,titles_prob_not)]

#The order "som" "iss" "not"
margin_proba = [titles_r.value_counts()[i]/titles_r.shape[0] for i in range(3)]
weighed_auc_titles = [margin_proba[i]*bin_auc[i] for i in range(3)]
total_auc_titles = sum(weighed_auc_titles)

print("Naive Bayes Title AUC weighed by class probabilites:",total_auc_titles)

#Now lets do the same for summaries.

#Lets fit the whole title dataset
nb.fit(summaries,summaries_r)

#prob of each category and the order
pp_summaries = nb.predict_proba(summaries)
order_summaries = nb.classes_

#call the functions.
summaries_prob_iss = collapse_proba("iss",pp_summaries,order_summaries)
summaries_resp_iss = collapse_response("iss",summaries_r)

summaries_prob_som = collapse_proba("som",pp_summaries,order_summaries)
summaries_resp_som = collapse_response("som",summaries_r)

summaries_prob_not = collapse_proba("not",pp_summaries,order_summaries)
summaries_resp_not = collapse_response("not",summaries_r)

#get false proba and true probs.
fpr_iss, tpr_iss, th1 = roc_curve(summaries_resp_iss,summaries_prob_iss)
fpr_som, tpr_som, th2 = roc_curve(summaries_resp_som,summaries_prob_som)
fpr_not, tpr_not, th3 = roc_curve(summaries_resp_not,summaries_prob_not)

#set up the plot.

f2 = plt.figure(2)
plt.plot(fpr_iss,tpr_iss,label="Controversial")
plt.plot(fpr_som,tpr_som,label="Somewhat Controversial")
plt.plot(fpr_not,tpr_not,label="Not Controversial")

#and lets go ahead and plot it:
plt.legend(loc="lower right")
plt.title("Naive Bayes Summary Binary Relevance ROC")
#f2.show()

#And lets get an AUC score. We weigh each individual AUC by
#class probabilies.

bin_auc = [roc_auc_score(summaries_resp_som,summaries_prob_som), \
          + roc_auc_score(summaries_resp_iss,summaries_prob_iss), \
          + roc_auc_score(summaries_resp_not,summaries_prob_not)]

#The order "som" "iss" "not"
margin_proba = [summaries_r.value_counts()[i]/summaries_r.shape[0] for i in range(3)]
weighed_auc_summaries = [margin_proba[i]*bin_auc[i] for i in range(3)]
total_auc_summaries = sum(weighed_auc_summaries)

print("Naive Bayes Summary AUC weighed by class probabilites:",total_auc_summaries)
print(bin_auc)

#Lets explore a bit