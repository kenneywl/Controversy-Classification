import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold


#Lets import our data:
summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

#####################################################################################
#ten fold cross validation returns accuracy.

def svm_cross(factors,response,model):
	factors_scaled = RobustScaler().fit_transform(factors)
	kf = KFold(n_splits=10)
	kf.get_n_splits(factors_scaled)

	acc = []
	for i,j in kf.split(factors_scaled):
		model.fit(factors_scaled[i,:],response[i])
		pred = model.predict(factors_scaled[j,:])
		acc += [accuracy_score(response[j],pred)]
	acc_m = sum(acc)/10
	return(acc_m)


#On titles:
print("On Titles:")
model = SVC(kernel="linear",gamma='scale')
print("Linear",svm_cross(titles,titles_r,model))

#.582548
model = SVC(kernel="rbf",gamma='scale')
print("rbf",svm_cross(titles,titles_r,model))
#.565667

model = SVC(kernel="sigmoid",gamma='scale')
print("Sigmoid",svm_cross(titles,titles_r,model))
#.493982

model = SVC(kernel="poly",gamma='scale')
print("Poly",svm_cross(titles,titles_r,model))
#.547877

#On summaries:
print("On Summaries:")
model = SVC(kernel="linear",gamma='scale')
print("Linear",svm_cross(summaries,summaries_r,model))
#.7

model = SVC(kernel="rbf",gamma='scale')
print("rbf",svm_cross(summaries,summaries_r,model))
#.669

model = SVC(kernel="sigmoid",gamma='scale')
print("Sigmoid",svm_cross(summaries,summaries_r,model))
#.57978

model = SVC(kernel="poly",gamma='scale')
print("Poly",svm_cross(summaries,summaries_r,model))
#.6638
