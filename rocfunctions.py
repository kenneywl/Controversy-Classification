import pandas as pd
from scipy.integrate import cumtrapz
import numpy as np

#The below function takes class probabilites and actual probs
#and the category you are interested in and makes the roc data points.

def cat_ROC(cat, pred_prob, order, actual_cat):

	order = list(order)
	pred_ind = order.index(cat)

	cat_list = list(actual_cat.value_counts().index)
	cat_ind = cat_list.index(cat)

	count = actual_cat.value_counts()

	acc_m = []
	for j in range(-1,102):
		iss_li = []
		for i in range(pred_prob.shape[0]):
			if pred_prob[i,pred_ind] >= j/100:
				iss_li += [cat]
			else:
				iss_li += ["not " + cat]

		truep_rate = 0
		falsep_rate = 0
		for i in range(pred_prob.shape[0]):
			is_iss = iss_li[i] == cat

			is_iss2 = actual_cat[i] == cat
			if is_iss and is_iss2:
 				truep_rate += 1

			not_iss2 = actual_cat[i] != cat
			if is_iss and not_iss2:
				falsep_rate += 1

		truep_rate /= count[cat_ind]
		falsep_rate /= sum(count) - count[cat_ind]

		acc_m += [(falsep_rate,truep_rate)]

	acc_m = list(set(acc_m))
	acc_m = sorted(acc_m,key=lambda x: x[0])
	acc_m = sorted(acc_m,key=lambda x: x[1])

	acc_m = pd.Series(acc_m)
	acc_m.name = cat

	return(acc_m)

#This returns the cumlative area under the curve from the data points above.
def cumarea(data):
	x = [i[0] for i in data]
	y = [i[1] for i in data]

	auc_v = list(cumtrapz(y,x))
	return(auc_v)

#This weighes each according to class probability.
def weighedAUC(cat1,cat2,cat3,actual_cat):
	names = [cat1.name,cat2.name,cat3.name]
	cat_list = list(actual_cat.value_counts().index)

	aucs = [max(cumarea(cat1)),max(cumarea(cat2)),max(cumarea(cat3))]
	aucs_weighed = [aucs[i]*actual_cat.value_counts().loc[names[i]]/actual_cat.shape[0] for i in range(3)]
	return(sum(aucs_weighed))