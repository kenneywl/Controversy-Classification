import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, GridSearchCV


#Lets import our data:
summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

#####################################################################################

#On titles:
# print("On Titles:")
# model = SVC(kernel="linear")

# Cs = [.014,.015,.016]
# para_grid = {"C" : Cs}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False)

# titles_trans = RobustScaler().fit_transform(titles)
# gridS.fit(titles_trans,titles_r)

# pd.set_option('display.width', 170)

# pd.set_option('display.max_columns', 500)

# print(gridS.best_params_)
# print(pd.DataFrame(gridS.cv_results_)[["mean_test_score","std_test_score","rank_test_score"]])

#On summaries:

# print("On summaries:")
# model = SVC(kernel="linear")

# Cs = [.255,.26,.265,.27,.275,.28]
# para_grid = {"C" : Cs}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False)

# summaries_trans = RobustScaler().fit_transform(summaries)
# gridS.fit(summaries_trans,summaries_r)

# pd.set_option('display.width', 170)

# pd.set_option('display.max_columns', 500)

# print(gridS.best_params_)
# print(pd.DataFrame(gridS.cv_results_)[["param_C","mean_test_score","std_test_score","rank_test_score"]])

########################
#On titles:
print("On Titles:")
model = SVC(kernel="rbf")

Cs = [14,15,16]
gamma = [.0004,.0005,.0006]
para_grid = {"C" : Cs,"gamma":gamma}
gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False)

titles_trans = RobustScaler().fit_transform(titles)
gridS.fit(titles_trans,titles_r)

pd.set_option('display.width', 170)

pd.set_option('display.max_columns', 500)

print(gridS.best_params_)
params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
print(params)





# def svm_cross(factors,response,model):
# 	factors_scaled = RobustScaler().fit_transform(factors)
# 	kf = KFold(n_splits=10)
# 	kf.get_n_splits(factors_scaled)

# 	acc = []
# 	for i,j in kf.split(factors_scaled):
# 		model.fit(factors_scaled[i,:],response[i])
# 		pred = model.predict(factors_scaled[j,:])
# 		acc += [accuracy_score(response[j],pred)]
# 	acc_m = sum(acc)/10
# 	return(acc_m)

# #.582548
# model = SVC(kernel="rbf",gamma='scale')
# print("rbf",svm_cross(titles,titles_r,model))
# #.565667

# model = SVC(kernel="sigmoid",gamma='scale')
# print("Sigmoid",svm_cross(titles,titles_r,model))
# #.493982

# model = SVC(kernel="poly",gamma='scale')
# print("Poly",svm_cross(titles,titles_r,model))
# #.547877

# #On summaries:
# print("On Summaries:")
# model = SVC(kernel="linear",gamma='scale')
# print("Linear",svm_cross(summaries,summaries_r,model))
# #.7

# model = SVC(kernel="rbf",gamma='scale')
# print("rbf",svm_cross(summaries,summaries_r,model))
# #.669

# model = SVC(kernel="sigmoid",gamma='scale')
# print("Sigmoid",svm_cross(summaries,summaries_r,model))
# #.57978

# model = SVC(kernel="poly",gamma='scale')
# print("Poly",svm_cross(summaries,summaries_r,model))
# #.6638
