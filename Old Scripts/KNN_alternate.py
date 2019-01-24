



# from sklearn.neighbors import KNeighborsClassifier
# print("On KNN Summaries:")

# model = KNeighborsClassifier()
# nn = [i for i in range(1,101)]
# para_grid =  {"n_neighbors" : nn}

# def custom_roc_auc_score(y_true,y_pred):
# 	bin_response = label_binarize(y_true, classes=list(np.unique(y_true)))
# 	auc = roc_auc_score(bin_response,y_pred,average="weighted")
# 	return(auc)

# scoring = {'acc': make_scorer(accuracy_score), 'auc': make_scorer(custom_roc_auc_score,needs_proba=True)}
# gridS = GridSearchCV(model, para_grid, scoring=scoring, cv=10, return_train_score=False, iid=False, refit=False)
# gridS.fit(summaries_trans,summaries_r)

# params = pd.DataFrame(gridS.cv_results_)[["param_n_neighbors","mean_test_acc","rank_test_acc","mean_test_auc","rank_test_auc"]]

# print(params)