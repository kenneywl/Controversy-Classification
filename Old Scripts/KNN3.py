import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold

#homebrew for plotting.

#from KNN_plot_func import *

#This gets accuracy and weighed AUC for all values of up to max k. whih_k calls auc_acc to get the values for that
#value of k.

def which_k(factors,response,max_k=30):
	auc = []
	acc = []
	for i in range(1,max_k+1):
		a, b = auc_acc_cross(factors,response,i)
		auc += [a]
		acc += [b]

	auc_max = max(auc)
	acc_max = max(acc)
	k_auc_max = auc.index(auc_max)+1
	k_acc_max = acc.index(acc_max)+1

	print(" AUC max info:", k_auc_max, auc_max,"\n","Accuracy max info:", k_acc_max, acc_max)

def auc_acc_cross(factors,response,k):
	kn = KNeighborsClassifier(n_neighbors=k)
	kf = KFold(n_splits=10)
	kf.get_n_splits(factors)

	acc = []
	auc = []
	for i,j in kf.split(factors):
		kn.fit(factors.iloc[i,:],response.iloc[i])

		#the stuffs for accuracy
		pred_response = kn.predict(factors.iloc[j,:])
		pred_accuracy = accuracy_score(response.iloc[j],pred_response)
		acc += [pred_accuracy]

		#Now for auc
		pred_probability = kn.predict_proba(factors.iloc[j,:])
		pred_probability = pd.DataFrame(pred_probability,columns=list(kn.classes_))

		bin_response = label_binarize(response.iloc[j], classes=list(kn.classes_))
		pred_auc = roc_auc_score(bin_response,pred_probability,average="weighted")
		auc += [pred_auc]

	#for acc
	acc_m = sum(acc)/10
	#for auc
	auc_m = sum(auc)/10
	return(auc_m,acc_m)