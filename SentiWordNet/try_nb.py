import pandas as pd
import statsmodels.api as sm
import numpy as np
from plot_confusion_matrix import *
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score


from sklearn.naive_bayes import GaussianNB

pd.options.display.float_format = '{:20,.3f}'.format

train_ind = pd.read_pickle("train_ind.pkl").tolist()
test_ind = pd.read_pickle("test_ind.pkl").tolist()

x = pd.read_pickle('art_m.pkl')
y = x.iloc[:,7]


def upper(data):
	if data == 'controversial':
		ans = 'Controversial'
	else:
		ans = 'Not Controversial'
	return ans

y = y.apply(upper)


def remove_error(data):
	no_error = []
	for i in data.index:
		if not isinstance(data['Neg Score'].loc[i], str):
			no_error += [i]
	return no_error

train_noerror_loc = remove_error(x.iloc[train_ind,:])
x_train = x.loc[train_noerror_loc,:]
y_train = y.loc[train_noerror_loc]

test_noerror_loc = remove_error(x.iloc[test_ind,:])
x_test = x.loc[test_noerror_loc,:]
y_true = y.loc[test_noerror_loc]

cols = ["Pos Score", "Neg Score","P/N Metric","Word Count"]

x_train = x_train.loc[:,cols]
x_test = x_test.loc[:,cols]

y_train_z = []
for i in y_train:
	if i == "Controversial":
		y_train_z += [1]
	else:
		y_train_z += [0]

y_true_z = []
for i in y_true:
	if i == "Controversial":
		y_true_z += [1]
	else:
		y_true_z += [0]

y_train = pd.Series(y_train_z,index=y_train.index)
y_true = pd.Series(y_true_z,index=y_true.index)

x_train.astype(float)

nb = GaussianNB()

nb.fit(x_train,y_train)


y_proba_zipped = nb.predict_proba(x_test)
_, y_proba = zip(*y_proba_zipped)
y_proba = pd.Series(y_proba)

from scipy.integrate import quad
from scipy.stats import beta

def average_meterics(x):
	pred = np.array(y_proba > x, dtype=int)
	a,b,c,d = 0,0,0,0
	for i,j in zip(pred,y_true):
		if i==0 and j==0:
			a += 1
		if i==0 and j==1:
			b += 1
		if i==1 and j==0:
			c += 1
		if i==1 and j==1:
			d += 1

	bb = beta.pdf(x,2,2)
	ans = bb*b
	return(ans)

ss, _ = quad(average_meterics,0,1,limit=100)
print(ss)