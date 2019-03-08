import pandas as pd
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from random import shuffle
from plot_confusion_matrix import *
import matplotlib.pyplot as plt
import statsmodels.api as sm

train_ind = pd.read_pickle("train_ind.pkl")
test_ind = pd.read_pickle("test_ind.pkl")

model = Doc2Vec.load("doument_model.d2v")


x = pd.DataFrame(columns=range(100))
y = pd.Series()
for i in model.docvecs.doctags:
	resp_split = i.split("_")
	index = int(resp_split[1])
	resp = resp_split[0]

	x.loc[index] = model[i]
	y.loc[index] = resp

# model = LogisticRegression(solver='liblinear')

# Cs = [.01,.012,.014,.016,.018,.02]
# para_grid = {"C" : Cs}
# gridS = GridSearchCV(model, para_grid, cv=10, return_train_score=False,iid=False)


# gridS.fit(x,y)

# print(gridS.best_params_,"ACC:",gridS.best_score_)

for i in y.index:
	if y.loc[i] == "controversial":
		y.loc[i] = 1
	else:
		y.loc[i] = 0


x_train = x.iloc[train_ind,:]
y_train = y.iloc[train_ind]

x_test = x.iloc[test_ind,:]
y_true = y.iloc[test_ind]

fit = sm.Logit(y_train,x_train).fit()

print(fit.summary())








# model = LogisticRegression(solver='liblinear',C=.016)

# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_proba = model.predict_proba(x_test)[:,1]

# acc = accuracy_score(y_true,y_pred)
# print(acc)

# plot_confusion_matrix(y_true=y_true,y_pred=y_pred,classes=model.classes_,title="Document Embeddings \n Logistic Regression Confusion Matrix")

# roc_average = roc_auc_score(y_true,y_proba)
# print(roc_average)


# fpr, tpr, _ = roc_curve(y_true,y_proba,pos_label='not controversial')


# f1 = plt.figure(1)
# plt.plot(fpr,tpr)
# plt.title("Document Embeddings \n Logistic Regression ROC Curve")
# plt.xlabel("True Positive Rate")
# plt.ylabel("False Positive Rate")
# plt.show()