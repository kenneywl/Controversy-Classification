import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold

#This gets accuracy and weighed AUC for all values of up to max k. which_k calls auc_acc to get the values for that
#value of k.

def knn_which_k(factors,response,max_k=30):
	meterics = []
	for i in range(1,max_k+1):
		meterics += [auc_acc_cross(factors,response,i)]

	auc, acc = zip(*meterics)

	auc_max = max(auc)
	acc_max = max(acc)
	k_auc_max = auc.index(auc_max)+1
	k_acc_max = acc.index(acc_max)+1

	print(" K checked up to: ",max_k,"\n AUC max info:", k_auc_max, auc_max,"\n","Accuracy max info:", k_acc_max, acc_max)

def auc_acc_cross(factors,response,k):
	kn = KNeighborsClassifier(n_neighbors=k)
	kf = KFold(n_splits=10)
	kf.get_n_splits(factors)

	#for acc
	pred_response = np.ndarray(shape=(0,))
	#for auc
	pred_probability = np.ndarray(shape=(0,3))
	for i,j in kf.split(factors):
		kn.fit(factors.iloc[i,:],response.iloc[i])

		pv = kn.predict(factors.iloc[j,:]) #predict the outcome
		pp = kn.predict_proba(factors.iloc[j,:]) #predict the probabilities

		pred_response = np.append(pred_response,pv,axis=0) #apppend to the master list
		pred_probability = np.append(pred_probability,pp,axis=0) #append to the master list

	#for acc
	pred_accuracy = accuracy_score(response,pred_response)
	#for auc
	pred_probability = pd.DataFrame(pred_probability,columns=list(kn.classes_))
	bin_response = label_binarize(response, classes=list(kn.classes_))
	pred_auc = roc_auc_score(bin_response,pred_probability,average="weighted")
	return(pred_auc,pred_accuracy)