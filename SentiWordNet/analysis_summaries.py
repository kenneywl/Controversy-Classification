import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

summaries_m = pd.read_pickle("summaries_m.pkl")

cont = []
ncont = []
scont = []
for i in range(summaries_m.shape[0]):
	if summaries_m['Response'].iloc[i] == "controversial":
		cont += [i]
	elif summaries_m['Response'].iloc[i] == "not controversial":
		ncont += [i]
	else:
		scont += [i]

scontroversial = summaries_m.iloc[scont]
controversial = summaries_m.iloc[cont]
ncontroversial = summaries_m.iloc[ncont]

to_keep = []
for i in controversial.index:
	if not np.isnan(controversial['Pos Score'].loc[i]):
		to_keep += [i]

controversial = controversial.loc[to_keep]

to_keep = []
for i in controversial.index:
	if not np.isnan(controversial['Neg Score'].loc[i]):
		to_keep += [i]

controversial = controversial.loc[to_keep]

to_keep = []
for i in ncontroversial.index:
	if not np.isnan(ncontroversial['Pos Score'].loc[i]):
		to_keep += [i]

ncontroversial = ncontroversial.loc[to_keep]

to_keep = []
for i in ncontroversial.index:
	if not np.isnan(ncontroversial['Neg Score'].loc[i]):
		to_keep += [i]

ncontroversial = ncontroversial.loc[to_keep]

##############################################################
#Method of moments estimates.

# sample_cont_total = 0
# for i in controversial['Pr(Response)']:
# 	sample_cont_total += i

# sample_cont_each = [i/sample_cont_total for i in controversial['Pr(Response)']]

# mean_cont_pos = sum([controversial['Pos Score'].iloc[i]*sample_cont_each[i]  for i in range(controversial.shape[0])])
# mean_cont_neg = sum([controversial['Neg Score'].iloc[i]*sample_cont_each[i]  for i in range(controversial.shape[0])])

# var_cont_pos = sum([(controversial['Pos Score'].iloc[i]-mean_cont_pos)**2*sample_cont_each[i]  for i in range(controversial.shape[0])])
# var_cont_neg = sum([(controversial['Neg Score'].iloc[i]-mean_cont_pos)**2*sample_cont_each[i]  for i in range(controversial.shape[0])])

# print(mean_cont_pos,mean_cont_neg)
# print(sqrt(var_cont_pos),sqrt(var_cont_neg))
# print(controversial['Pos Score'].describe())
# print(controversial['Neg Score'].describe())

# alpha_cont = mean_cont_pos*(mean_cont_pos*(1-mean_cont_pos)/var_cont_neg-1)
# beta_cont = (1-mean_cont_pos)*(mean_cont_pos*(1-mean_cont_pos)/var_cont_neg-1)


# nsample_cont_total = 0
# for i in ncontroversial['Pr(Response)']:
# 	nsample_cont_total += i

# nsample_cont_each = [i/nsample_cont_total for i in ncontroversial['Pr(Response)']]

# mean_ncont_pos = sum([ncontroversial['Pos Score'].iloc[i]*nsample_cont_each[i]  for i in range(ncontroversial.shape[0])])
# mean_ncont_neg = sum([ncontroversial['Neg Score'].iloc[i]*nsample_cont_each[i]  for i in range(ncontroversial.shape[0])])

# var_ncont_pos = sum([(ncontroversial['Pos Score'].iloc[i]-mean_cont_pos)**2*nsample_cont_each[i]  for i in range(ncontroversial.shape[0])])
# var_ncont_neg = sum([(ncontroversial['Neg Score'].iloc[i]-mean_cont_pos)**2*nsample_cont_each[i]  for i in range(ncontroversial.shape[0])])

# print(mean_ncont_pos,mean_ncont_neg)
# print(sqrt(var_ncont_pos),sqrt(var_ncont_neg))
# print(ncontroversial['Pos Score'].describe())
# print(ncontroversial['Neg Score'].describe())

# alpha_ncont = mean_ncont_pos*(mean_ncont_pos*(1-mean_ncont_pos)/var_ncont_neg-1)
# beta_ncont = (1-mean_ncont_pos)*(mean_ncont_pos*(1-mean_ncont_pos)/var_ncont_neg-1)


# print("Controversial")
# print("Alpha:", alpha_cont,"Beta:", beta_cont)

# print("Not Controversial")
# print("Alpha:", alpha_ncont,"Beta:", beta_ncont)

# Controversial
# Alpha: 2.383106946816734 Beta: 59.75694986379279
# Not Controversial
# Alpha: 2.9015528551577003 Beta: 62.27976460403675

# print(controversial.shape[0],ncontroversial.shape[0])





# #Are pos and negative correlated?
f1 = plt.figure(1)
plt.scatter(ncontroversial['Neg Score'],ncontroversial['Pos Score'],label="Not Controversial")
plt.scatter(controversial['Neg Score'],controversial['Pos Score'],label="Controversial")
plt.xlabel("Negative Score")
plt.ylabel("Positive Score")
plt.legend(loc="lower right")
plt.title("Pos Score by Neg Score: Independant?")
f1.show()
# # #Looks like: no, they are not. (Thats good!)

# # # #
# f2 = plt.figure(2)
# plt.hist(ncontroversial['M Metric'],label="Not Controversial",density=True)
# plt.hist(controversial['M Metric'],label="Controversial",density=True)
# plt.title("Multiplication Metric")
# plt.legend()
# f2.show()

# print(ncontroversial['M Metric'].describe())
# print(controversial['M Metric'].describe())

# f3 = plt.figure(3)
# plt.hist(controversial['Div Metric'],label="Controversial",density=True)
# plt.hist(ncontroversial['Div Metric'],label="Not Controversial",density=True)
# plt.title("Division Metric")
# plt.legend()
# f3.show()

# print(ncontroversial['Div Metric'].describe())
# print(controversial['Div Metric'].describe())

# f3 = plt.figure(3)
# plt.hist(ncontroversial['Relative Entropy'],label="Not Controversial",density=True)
# plt.hist(controversial['Relative Entropy'],label="Controversial",density=True)
# plt.title("Relative Entropy Metric")
# plt.legend()
# f3.show()

# print(ncontroversial['Relative Entropy'].describe())
# print(controversial['Relative Entropy'].describe())



# # f4 = plt.figure(4)
# # plt.hist(ncontroversial['A Metric'],label="Not Controversial",density=True)
# # plt.hist(controversial['A Metric'],label="Controversial",density=True)
# # plt.title("A Metric")
# # plt.legend()
# # f4.show()

# f5 = plt.figure(5)
# plt.hist(ncontroversial['Abs Metric'],label="Not Controversial",density=True)
# plt.hist(controversial['Abs Metric'],label="Controversial",density=True)
# plt.title("Abs Metric")
# plt.legend()
# f5.show()

# print(ncontroversial['Abs Metric'].describe())
# print(controversial['Abs Metric'].describe())

