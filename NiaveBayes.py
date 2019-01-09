import cleandata as cd

#for niave bayes:
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

def cross_NB(factors,response):
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
	return(acc)

#for the titles.
# cc = cross_NB(cd.titles,cd.titles_r)
# cc_av = sum(cc)/10
# print(cc,"\nAverage Accuracy:",cc_av)

#Average acc is .569

#for the summaries:
# cc = cross_NB(cd.summaries,cd.summaries_r)
# cc_av = sum(cc)/10
# print(cc,"\nAverage Accuracy:",cc_av)

#.648 accuracy.

nb = GaussianNB()
nb.fit(cd.summaries,cd.summaries_r)
ypred = nb.predict_proba(cd.summaries)
print(ypred)
