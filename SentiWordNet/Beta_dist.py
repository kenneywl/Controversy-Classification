# from scipy.stats import beta
# from scipy.integrate import quad
# from numpy import linspace
# import matplotlib.pyplot as plt
# import pandas as pd

#priors: cont: 118 ncont: 598

# Controversial
# Alpha: 2.383106946816734 Beta: 59.75694986379279
# Not Controversial
# Alpha: 2.9015528551577003 Beta: 62.27976460403675

# cont_alpha = 2.383106946816734
# cont_beta = 59.75694986379279
# cont_prior = 118/(118+598)


# ncont_alpha = 2.9015528551577003
# ncont_beta = 62.27976460403675
# ncont_prior = 598/(598+118)

# def prob_cont(x):
# 	top = beta.pdf(x,cont_alpha,cont_beta)*cont_prior
# 	bottom = top+beta.pdf(x,ncont_alpha,ncont_beta)*ncont_prior
# 	return(top/bottom)

# x = linspace(.0001,.9999,1000)
# p_c = [prob_cont(i) for i in x]
# p_nc = [1-i for i in p_c]

# plt.plot(x,p_c,label="Probability of Controversial")
# plt.plot(x,p_nc,label="Probability of Not Controversial")
# plt.hlines(ncont_prior,xmin=0,xmax=1,label="Not Controversial Prior")
# plt.hlines(cont_prior,xmin=0,xmax=1,label="Controversial Prior")
# plt.legend()
# plt.title("Simple Baysian Estimates")
# plt.show()


# art = pd.read_pickle('art_m.pkl')

# no_error = []
# for i in art.index:
# 	if not isinstance(art.loc[i,'P/N Metric'], str):
# 		no_error += [i]

# art = art.loc[no_error,:]

# def alpha_out(data):
# 	quad.pdf(data,2,2)


# import pandas as pd
# import statsmodels.api as sm
# import numpy as np
# from plot_confusion_matrix import *
# from sklearn.metrics import accuracy_score

# pd.options.display.float_format = '{:20,.3f}'.format

# train_ind = pd.read_pickle("train_ind.pkl").tolist()
# test_ind = pd.read_pickle("test_ind.pkl").tolist()

# x = pd.read_pickle('art_m.pkl')
# y = x.iloc[:,7]


# def upper(data):
# 	if data == 'controversial':
# 		ans = 'Controversial'
# 	else:
# 		ans = 'Not Controversial'
# 	return ans

# y = y.apply(upper)


# def remove_error(data):
# 	no_error = []
# 	for i in data.index:
# 		if not isinstance(data['Neg Score'].loc[i], str):
# 			no_error += [i]
# 	return no_error

# # train_noerror_loc = remove_error(x.iloc[train_ind,:])
# # x_train = x.loc[train_noerror_loc,:]
# # y_train = y.loc[train_noerror_loc]

# # test_noerror_loc = remove_error(x.iloc[test_ind,:])
# # x_test = x.loc[test_noerror_loc,:]
# # y_true = y.loc[test_noerror_loc]
# # y_pred = []

# train_noerror_loc = remove_error(x)
# x_test = x.loc[train_noerror_loc,:]
# y_true = y.loc[train_noerror_loc]


# y_pred = []
# for i in x_test.index:
# 	if x_test.loc[i,'P/N Metric'] < 0:
# 		y_pred += [1]
# 	else:
# 		y_pred += [0]

# y_true_z = []
# for i in y_true:
# 	if i == "Controversial":
# 		y_true_z += [1]
# 	else:
# 		y_true_z += [0]


# y_true_z = pd.Series(y_true_z,index=y_true.index)

# acc = accuracy_score(y_true_z,y_pred)
# print(acc)