import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer,roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import label_binarize

#Lets import our data:
summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

#####################################################################################
titles_trans = RobustScaler().fit_transform(titles)
summaries_trans = RobustScaler().fit_transform(summaries)

pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000) 

# #On titles: #.593 for c=.015
# print("On linear Titles:")
# model = SVC(kernel="linear")

# Cs = [.014,.015,.016]
# para_grid = {"C" : Cs}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


# gridS.fit(titles_trans,titles_r)

# print(gridS.best_params_,"ACC:",gridS.best_score_)
# # print(pd.DataFrame(gridS.cv_results_)[["param_C","mean_test_score","std_test_score","rank_test_score"]])

# #On summaries: .713 c=.27

# print("On linear Summaries:")
# model = SVC(kernel="linear")

# Cs = [.265,.27,.275]
# para_grid = {"C" : Cs}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False)


# gridS.fit(summaries_trans,summaries_r)

# print(gridS.best_params_,"ACC:",gridS.best_score_)
# # print(pd.DataFrame(gridS.cv_results_)[["param_C","mean_test_score","std_test_score","rank_test_score"]])

# #######################
# #On titles: .584 c= 15 gamma=.0004
# print("On rbf Titles:")
# model = SVC(kernel="rbf")

# Cs = [14,15,16]
# gamma = [.0004,.0005,.0006]
# para_grid = {"C" : Cs,"gamma":gamma}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


# gridS.fit(titles_trans,titles_r)

# print(gridS.best_params_,"ACC:",gridS.best_score_)
# # params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# # # print(params)

# #on summaries .718 for c= 17 and gamma = .0008
# print("On rbd Summaries:")
# model = SVC(kernel="rbf")

# Cs = [15,16,17]
# gamma = [.0008,.0009,.001]
# para_grid = {"C" : Cs,"gamma":gamma}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


# gridS.fit(summaries_trans,summaries_r)

# print(gridS.best_params_,"ACC:",gridS.best_score_)
# # params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# # print(params)

# ##########################################################################
# # sigmoid
# # On titles: .574 c=.8 gamma = .0007
# print("On sigmoid Titles:")
# model = SVC(kernel="sigmoid")

# Cs = [.8,.9,1]
# gamma = [.0005,.0006,.0007]
# para_grid = {"C" : Cs,"gamma":gamma}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


# gridS.fit(titles_trans,titles_r)


# print(gridS.best_params_,"ACC:",gridS.best_score_)
# # params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# # print(params)

# #on summaries 0.693 for c= 11 and gamma = .001
# print("On sigmoid Summaries:")
# model = SVC(kernel="sigmoid")

# Cs = [10,11,12]
# gamma = [.0005,.001,.005]
# para_grid = {"C" : Cs,"gamma":gamma}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False,refit=False)


# gridS.fit(summaries_trans,summaries_r)

# print(gridS.best_params_,"ACC:",gridS.best_score_)
# params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# print(params)

#This is the response:
# On linear Titles:
# {'C': 0.015} ACC: 0.592583858089476
# On linear Summaries:
# {'C': 0.265} ACC: 0.7130750154309111
# On rbf Titles:
# {'C': 15, 'gamma': 0.0004} ACC: 0.583620062833546
# On rbf Summaries:
# {'C': 17, 'gamma': 0.0008} ACC: 0.7182919120037567
# On sigmoid Titles:
# {'C': 0.8, 'gamma': 0.0007} ACC: 0.5735085264298747
# On sigmoid Summaries:
# {'C': 11, 'gamma': 0.001} ACC: 0.6926779737682364