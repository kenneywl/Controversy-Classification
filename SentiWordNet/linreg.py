import pandas as pd
import statsmodels.api as sm
import numpy as np

pd.options.display.float_format = '{:20,.3f}'.format

art = pd.read_pickle("art_all_m.pkl")
# su = pd.read_excel("Summaries.xlsx",nrows=1000)

# index = []
# for i in range(1000):
# 	cont = su.loc[i,'Classification'] == 'controversial'
# 	ncont = su.loc[i,'Classification'] == 'not controversial'
# 	if cont or ncont:
# 		index += [i]

# su = su.iloc[index]

# def prob(row):
# 	new = row['controversial']-row['not controversial']
# 	return(new)

# y = su.apply(prob,axis=1)


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

x['Response'] = x['Response'].apply(lambda x: 1/(1-np.exp(-x)))


############################
#analysis

cols = ["Neg Score","Pos Score","P+N Metric","Abs(P-N) Metric", "Word Count"]
x = x.loc[:,cols]
x = x.astype(float)


x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
print(model.summary())
