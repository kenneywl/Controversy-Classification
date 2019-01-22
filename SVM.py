import pandas as pd
import numpy as np
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

pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 500)

# #On titles: #.593 for c=.015
# print("On linear Titles:")
# model = SVC(kernel="linear")

# Cs = [.014,.015,.016]
# para_grid = {"C" : Cs}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False)

# titles_trans = RobustScaler().fit_transform(titles)
# gridS.fit(titles_trans,titles_r)

# print(gridS.best_params_)
# print(pd.DataFrame(gridS.cv_results_)[["param_C","mean_test_score","std_test_score","rank_test_score"]])

# #On summaries: .713 c=.27

# print("On linear Summaries:")
# model = SVC(kernel="linear")

# Cs = [.265,.27,.275]
# para_grid = {"C" : Cs}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False)

# summaries_trans = RobustScaler().fit_transform(summaries)
# gridS.fit(summaries_trans,summaries_r)

# print(gridS.best_params_)
# print(pd.DataFrame(gridS.cv_results_)[["param_C","mean_test_score","std_test_score","rank_test_score"]])

########################
# #On titles: .584 c= 15 gamma=.0004
# print("On rbf Titles:")
# model = SVC(kernel="rbf")

# Cs = [14,15,16]
# gamma = [.0004,.0005,.0006]
# para_grid = {"C" : Cs,"gamma":gamma}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False)

# titles_trans = RobustScaler().fit_transform(titles)
# gridS.fit(titles_trans,titles_r)

# print(gridS.best_params_)
# params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# print(params)

# #on summaries .718 for c= 17 and gamma = .0008
# print("On rbd Summaries:")
# model = SVC(kernel="rbf")

# Cs = [15,16,17]
# gamma = [.0008,.0009,.001]
# para_grid = {"C" : Cs,"gamma":gamma}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False)

# summaries_trans = RobustScaler().fit_transform(summaries)
# gridS.fit(summaries_trans,summaries_r)

# print(gridS.best_params_)
# params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# print(params)

###########################################################################
#sigmoid
#On titles: .574 c=.8 gamma = .0007
# print("On sigmoid Titles:")
# model = SVC(kernel="sigmoid")

# Cs = [.8,.9,1]
# gamma = [.0005,.0006,.0007]
# para_grid = {"C" : Cs,"gamma":gamma}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False)

# titles_trans = RobustScaler().fit_transform(titles)
# gridS.fit(titles_trans,titles_r)


# print(gridS.best_params_)
# params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# print(params)

# #on summaries 0.693 for c= 11 and gamma = .001
# print("On sigmoid Summaries:")
# model = SVC(kernel="sigmoid")

# Cs = [10,11,12]
# gamma = [.0005,.001,.005]
# para_grid = {"C" : Cs,"gamma":gamma}
# gridS = GridSearchCV(model, para_grid, cv=10,return_train_score=False,iid=False)

# summaries_trans = RobustScaler().fit_transform(summaries)
# gridS.fit(summaries_trans,summaries_r)

# print(gridS.best_params_)
# params = pd.DataFrame(gridS.cv_results_)[["param_C","param_gamma","mean_test_score","rank_test_score"]]
# print(params)
