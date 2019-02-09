import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer,roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import label_binarize

#Lets import our data:
# summaries = pd.read_pickle("summaries.pkl")
# summaries_r = pd.read_pickle("summaries_r.pkl")
# titles = pd.read_pickle("titles.pkl")
# titles_r = pd.read_pickle("titles_r.pkl")

agreements = pd.read_pickle("agreements.pkl")
agreements_r = pd.read_pickle("agreements_r.pkl")

#####################################################################################

# pd.set_option('display.width', 170)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_rows', 1000) 


# print("On linear Titles:")
# model = SVC(kernel="linear")

# Cs = [.014,.015,.016]
# para_grid = {"C" : Cs}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


# gridS.fit(titles,titles_r)

# print(gridS.best_params_,"ACC:",gridS.best_score_)
# # print(pd.DataFrame(gridS.cv_results_)[["param_C","mean_test_score","std_test_score","rank_test_score"]])

# print("On linear Summaries:")
# model = SVC(kernel="linear")

# Cs = [.27,.275,.28]
# para_grid = {"C" : Cs}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False)


# gridS.fit(summaries,summaries_r)

# print(gridS.best_params_,"ACC:",gridS.best_score_)
# #print(pd.DataFrame(gridS.cv_results_)[["param_C","mean_test_score","std_test_score","rank_test_score"]])
# #On summaries: .713 c=.27


# #######################

# print("On rbf Titles:")
# model = SVC(kernel="rbf")

# Cs = [15,16,17]
# gamma = [.0004,.0005,.0006]
# para_grid = {"C" : Cs,"gamma":gamma}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


# gridS.fit(titles,titles_r)

# print(gridS.best_params_,"ACC:",gridS.best_score_)
# #params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# #print(params)


# print("On rbf Summaries:")
# model = SVC(kernel="rbf")

# Cs = [16,17,18]
# gamma = [.0007,.0008,.0009]
# para_grid = {"C" : Cs,"gamma":gamma}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


# gridS.fit(summaries,summaries_r)

# print(gridS.best_params_,"ACC:",gridS.best_score_)
# #params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# #print(params)

# ##########################################################################
# # sigmoid
# print("On sigmoid Titles:")
# model = SVC(kernel="sigmoid")

# Cs = [.9,1,1.1]
# gamma = [.0006,.0007,.0008]
# para_grid = {"C" : Cs,"gamma":gamma}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


# gridS.fit(titles,titles_r)


# print(gridS.best_params_,"ACC:",gridS.best_score_)
# #params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# #print(params)

# #on summaries 0.693 for c= 11 and gamma = .001
# print("On sigmoid Summaries:")
# model = SVC(kernel="sigmoid")

# Cs = [10,11,12]
# gamma = [.0005,.001,.0015]
# para_grid = {"C" : Cs,"gamma":gamma}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


# gridS.fit(summaries,summaries_r)

# print(gridS.best_params_,"ACC:",gridS.best_score_)
# #params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# #print(params)

print("On linear Agreements:")
model = SVC(kernel="linear")

Cs = [.1] #c=.1 acc= .690
para_grid = {"C" : Cs}
gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


gridS.fit(agreements,agreements_r)

print(gridS.best_params_,"ACC:",gridS.best_score_)
# print(pd.DataFrame(gridS.cv_results_)[["param_C","mean_test_score","std_test_score","rank_test_score"]])

print("On rbf Agreements:")
model = SVC(kernel="rbf")

Cs = [6,7,8]
gamma = [.0003,.0004,.0005]
para_grid = {"C" : Cs,"gamma":gamma}
gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


gridS.fit(agreements,agreements_r)

print(gridS.best_params_,"ACC:",gridS.best_score_)
# params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# print(params)

#c=7,g=.004 acc=.659

print("On sigmoid Agreements:")
model = SVC(kernel="sigmoid")

Cs = [10,11,12]
gamma = [.0005,.001,.0015]
para_grid = {"C" : Cs,"gamma":gamma}
gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


gridS.fit(agreements,agreements_r)

print(gridS.best_params_,"ACC:",gridS.best_score_)
# params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# print(params)

#c=11,g=.001 acc=.562




#This is the response:
# On linear Titles:
# {'C': 0.015} ACC: 0.5768660054053314
# On linear Summaries:
# {'C': 0.275} ACC: 0.6918303128535432
# On rbf Titles:
# {'C': 16, 'gamma': 0.0004} ACC: 0.5768537953931212
# On rbd Summaries:
# {'C': 18, 'gamma': 0.0007} ACC: 0.6841890015690055
# On sigmoid Titles:
# {'C': 0.9, 'gamma': 0.0008} ACC: 0.5768793129467287
# On sigmoid Summaries:
# {'C': 11, 'gamma': 0.0015} ACC: 0.6661243571762506

# On linear Agreements:
# {'C': 0.1} ACC: 0.6904278793628889
# On rbf Agreements:
# {'C': 7, 'gamma': 0.0004} ACC: 0.659260357368754
# On sigmoid Agreements:
# {'C': 10, 'gamma': 0.0005} ACC: 0.5618746765568398