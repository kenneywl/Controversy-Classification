import pandas as pd
import statsmodels.api as sm
import numpy as np
from plot_confusion_matrix import *
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score
from sklearn.preprocessing import RobustScaler


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

# x_train_loc = remove_error(x)
# x_train = x.loc[x_train_loc,:]
# y_train = y.loc[x_train_loc]

###
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

#################################################################
#################################################################
#analysis

cols = ["Word Count","Neg Score","Pos Score",'P/N Metric']
x_train = x_train.loc[:,cols]
x_train = x_train.astype(float)

# x_train = pd.DataFrame(RobustScaler().fit_transform(x_train),columns=list(x_train),index=x_train.index)
intercept = [1.0 for i in range(x_train.shape[0])]
x_train = x_train.assign(Intercept=intercept)

logit = sm.Logit(y_train,x_train)
result = logit.fit(method = 'powell')
# print(result.summary(alpha=.05))
# print(np.exp(result.params))

# print(result.summary(alpha=.05).as_latex())

x_test = x_test.loc[:,cols]
x_test = x_test.astype(float)
# x_test = pd.DataFrame(RobustScaler().fit_transform(x_test),columns=list(x_test),index=x_test.index)
intercept = [1.0 for i in range(x_test.shape[0])]
x_test = x_test.assign(Intercept=intercept)
y_proba = result.predict(x_test)

# roc_average = roc_auc_score(y_true_z,y_proba)
# print(roc_average)

# y_pred = [0 if i < .415 else 1 for i in y_proba]
# plot_confusion_matrix(y_true=y_true,y_pred=y_pred, classes=["0","1"],title="Logistic Regression Confusion Matrix")


###################################################################################
###################################################################################
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

ss, _ = quad(average_meterics,0,1,limit=50)
print(ss)

# x = np.linspace(0,1,50)
# y = [average_meterics(i) for i in x]
# plt.plot(x,y)
# plt.show()

# print(average_precision(.25))

# fpr, tpr, _ = roc_curve(y_true,y_proba,pos_label='Controversial')

# f2 = plt.figure(6)
# plt.hist(y_proba)
# plt.show()

# f1 = plt.figure(5)
# plt.plot(fpr,tpr)
# plt.title("SentiWordNet \n Logistic Regression ROC Curve")
# plt.xlabel("True Positive Rate")
# plt.ylabel("False Positive Rate")
# plt.show()


# avprec = average_precision_score(y_true,y_proba,pos_label="Controversial")
# print(avprec)
