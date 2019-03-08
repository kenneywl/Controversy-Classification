import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from plot_confusion_matrix import *
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:20,.3f}'.format
pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000) 



su = pd.read_pickle("summaries.pkl")
su_r = pd.read_pickle("summaries_r.pkl")

train_ind = pd.read_pickle("train_ind.pkl")
test_ind = pd.read_pickle("test_ind.pkl")

index = []
for i in range(940):
	cont = su_r.iloc[i] == 'Controversial'
	ncont = su_r.iloc[i] == 'Not Controversial'
	if cont or ncont:
		index += [i]

y = su_r.iloc[index]
x = su.iloc[index,:]

x_train = x.iloc[train_ind,:]
y_train = y.iloc[train_ind]

x_test = x.iloc[test_ind,:]
y_true = y.iloc[test_ind]

# model = LogisticRegression(solver='liblinear')

# Cs = [.6,.8,1,1.2,1.4]
# para_grid = {"C" : Cs}
# gridS = GridSearchCV(model, para_grid, cv=10, return_train_score=False,iid=False)

# gridS.fit(x,y)

# print(gridS.best_params_,"ACC:",gridS.best_score_)
#c = .8 is best

# model = LogisticRegression(solver='liblinear',C=.8)

# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_proba = model.predict_proba(x_test)[:,1]

# acc = accuracy_score(y_true,y_pred)
# print(acc)

# plot_confusion_matrix(y_true=y_true,y_pred=y_pred,classes=model.classes_,title="LWIC \n Logistic Regression Confusion Matrix")

# roc_average = roc_auc_score(y_true,y_proba)
# print(roc_average)

# fpr, tpr, _ = roc_curve(y_true,y_proba,pos_label='Not Controversial')


# f1 = plt.figure(1)
# plt.plot(fpr,tpr)
# plt.title("LWIC \n Logistic Regression ROC Curve")
# plt.xlabel("True Positive Rate")
# plt.ylabel("False Positive Rate")
# plt.show()


#no dim reducation
# 0.8275862068965517
# 0.8808539944903582

#with dim reduction

# 0.8620689655172413
# 0.8860192837465565

# print(model.coef_)
# print(model.decision_function(x_test))
# print(model.intercept_)


####################################################################
import statsmodels.api as sm
import numpy as np

y_train_z = []
for i in y_train:
	if i == "Not Controversial":
		y_train_z += [0]
	else:
		y_train_z += [1]

y_train_z = pd.Series(y_train_z,index=y_train.index)

intercept = [1.0 for i in range(x.shape[0])]
x_train = x_train.assign(intercept=intercept)

logit = sm.Logit(y_train_z,x_train)
result = logit.fit(method = 'powell')

ranks = result.pvalues.rank().astype(int).apply(lambda x: x-1).tolist()
sums = result.summary2().tables[1]

rank_in = []
for i in range(len(ranks)):
	rank_in += [(i,ranks[i])]

rank_in.sort(key=lambda x: x[1])

pr_rank, _ = zip(*rank_in)
pr_rank = list(pr_rank)
result2 = sm.Logit(y_train_z,x_train.iloc[:,pr_rank]).fit(method = 'powell')
print(result2.summary())


# intercept = [1.0 for i in range(y_true.shape[0])]
# x_test = x_test.assign(intercept=intercept)
# y_proba = result.predict(x_test)
# y_pred = ["Not Controversial" if i < .5 else "Controversial" for i in y_proba]

# plot_confusion_matrix(y_true=y_true,y_pred=y_pred, classes= ["Controverisal","Not Controverisal"],title="LWIC \n Logistic Regression Confusion Matrix")

# roc_average = roc_auc_score(y_true,[1-i for i in y_proba])
# print(roc_average)

# fpr, tpr, _ = roc_curve(y_true,y_proba,pos_label='Controversial')


# f1 = plt.figure(1)
# plt.plot(fpr,tpr)
# plt.title("LWIC \n Logistic Regression ROC Curve")
# plt.xlabel("True Positive Rate")
# plt.ylabel("False Positive Rate")
# plt.show()
