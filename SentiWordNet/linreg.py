import pandas as pd
import statsmodels.api as sm
import numpy as np
import statsmodels.formula.api as smf

pd.options.display.float_format = '{:20,.3f}'.format

art = pd.read_pickle("art_m.pkl")
su = pd.read_excel("Summaries.xlsx",nrows=1000)

index = []
for i in range(1000):
	cont = su.loc[i,'Classification'] == 'controversial'
	ncont = su.loc[i,'Classification'] == 'not controversial'
	if cont or ncont:
		index += [i]

su = su.iloc[index]

def prob(row):
	margin = row['controversial']+row['not controversial']
	new = row['controversial']/margin
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
x = sm.add_constant(x)

y = y.astype(float).loc[clean_loc]

############################
#analysis

cols = ["const", "Pos Score","Neg Score","Word Count","P/N Metric"]
x = x.loc[:,cols]

x = x.astype(float)

model = sm.OLS(y,x).fit()
print(model.summary())
