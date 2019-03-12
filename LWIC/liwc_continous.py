import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

pd.options.display.float_format = '{:20,.3f}'.format
pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000) 

art = pd.read_pickle("art_idf_m.pkl")
cols = ["Neg_Score","Pos_Score","Absolute_Diff","PN_Metric"]#, "Word_Count"]
x = art.loc[:,cols]

fa = pd.read_excel("Factors.xlsx")

fa.columns = [x.lower() for x in fa.columns]
dim_red = ["power","negemo","anger","tone","drives","percept","comma","see","discrep",\
		 "auxverb","dic","allpunc","risk","otherp","anx","function","pronoun","ppron",\
		  "cogproc","number","clout","negate","social","leisure","we","analytic",\
		  "differ","affect","achieve","certain","sad","tentat","work","prep","wc",\
		  "reward","wps","space","article","verb","hear","period","affiliation","relativ"]

col_names = [i for i in fa.columns if i in dim_red]
fa.index = x.index
fa = fa.loc[:,col_names]
x = x.join(fa)

y = art.loc[:,'Response']
x = sm.add_constant(x)
x = x.astype(float)
result = sm.OLS(y,x).fit()

ranks = result.pvalues.rank().astype(int).apply(lambda x: x-1).tolist()

rank_in = []
for i in range(len(ranks)):
	rank_in += [(i,ranks[i])]

rank_in.sort(key=lambda x: x[1])


pr_rank, _ = zip(*rank_in)
pr_rank = list(pr_rank)
x = x.iloc[:,pr_rank]
ncols = sum(np.array(result.pvalues < .6,dtype=int))
print(ncols)
x = x.iloc[:,:ncols]
result2 = sm.OLS(y,x).fit()
print(result2.summary())