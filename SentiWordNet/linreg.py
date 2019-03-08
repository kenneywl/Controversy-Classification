import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.api as smg

pd.options.display.float_format = '{:20,.3f}'.format
pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000) 


art = pd.read_pickle("art_all_m.pkl")
su = pd.read_excel("Summaries.xlsx",nrows=1000)


def prob(row):
	new = row['controversial']-row['not controversial']
	return(new)

y = su.apply(prob,axis=1)


def remove_error(data):
	no_error = []
	for i in data.index:
		if not isinstance(data['P/N Metric'].loc[i], str):
			no_error += [i]
	return no_error

clean_loc = remove_error(art)
x = art.loc[clean_loc,:]

y = x["Response"].astype(float).loc[clean_loc]

def abs_sum(row):
	return(abs(row["P-N Metric"]))

x["Abs(P-N) Metric"] = x.apply(abs_sum,axis=1)

# ############################
# #analysis

cols = ["Neg Score","Pos Score","P+N Metric","Abs(P-N) Metric", "Word Count"]
x = x.loc[:,cols]
x = x.astype(float)


x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
print(model.summary())

# coef = np.corrcoef(x.T)
# print(coef)
# smg.plot_corr_grid([coef]*6,xnames=list(x))
# plt.show()

from scipy.integrate import quad
from scipy.stats import beta

trans = model.fittedvalues.apply(lambda x: (x+20)/40)

def average_meterics(x):
	pred = np.array(trans > x, dtype=int)

	a,b,c,d = 0,0,0,0
	for i,j in zip(pred,y_true):
		if i==0 and j==0:
			a += 1
		if i==0 and j==1:
			b += 1
		if i==1 and j==0:
			c += 1
		if i==1 and j==1:
			d += 1

	bb = beta.pdf(x,2,2)
	ans = bb*b
	return(ans)

ss, _ = quad(average_meterics,0,1,limit=50)
print(ss)