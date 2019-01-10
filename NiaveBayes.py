import cleandata as cd
import pandas as pd

#for niave bayes:
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

def cross_NB(factors,response):

	factors = factors.sample(frac=1)
	response = response.reindex_like(factors)

	factors.index = range(len(factors))
	response.index = range(len(response))

	hold = int(factors.shape[0]/10)

	nb = GaussianNB()
	acc = []
	for i in range(10):
		ind = [j for j in range(hold*i,hold*(i+1))]

		set_p = factors.drop(ind)
		set_r = response.drop(ind)

		nb.fit(set_p,set_r)
		ypred = nb.predict(factors.iloc[ind,:])

		acc += [metrics.accuracy_score(response.iloc[ind],ypred)]

	print("10 Fold, Shuffled, Average Accuracy:",sum(acc)/10)

	return()

#for the titles.
# print("For the titles:")
# cross_NB(cd.titles,cd.titles_r)

#Average acc is .567
#Note this value changes slightly because 
#I randomly shuffle the rows.

#for the summaries:
# print("For the summaries:")
# cross_NB(cd.summaries,cd.summaries_r)

#.656 accuracy.
#Note this value changes slightly because 
#I randomly shuffle the rows.

#To do multiclass roc curve, I would have to pick an output to
#base it off of, do that?