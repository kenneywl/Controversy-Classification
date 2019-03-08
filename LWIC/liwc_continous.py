import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

pd.options.display.float_format = '{:20,.3f}'.format
pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000) 



fa = pd.read_excel("Factors.xlsx")

fa_fe = [i.lower() for i in list(fa)]
fa_ch = ["power","negemo","anger","tone","drives","percept","comma","see","discrep",\
		 "auxverb","dic","allpunc","risk","otherp","anx","function","pronoun","ppron",\
		  "cogproc","number","clout","negate","social","leisure","we","analytic",\
		  "differ","affect","achieve","certain","sad","tentat","work","prep","wc",\
		  "reward","wps","space","article","verb","hear","period","affiliation","relativ"]

ind2 = []
for i in fa_ch:
	for j in range(len(fa_fe)):
		if i == fa_fe[j]:
			ind2 += [j]

x = fa.iloc[:,ind2]
x = sm.add_constant(x)

su = pd.read_excel("Summaries.xlsx",nrows=1000)

def resp(row):
	a = row['controversial']
	b = row['not controversial']
	ans = a-b
	return(ans)

y = su.apply(resp,axis=1)

result = sm.OLS(y,x).fit()

ranks = result.pvalues.rank().astype(int).apply(lambda x: x-1).tolist()
sums = result.summary2().tables[1]

rank_in = []
for i in range(len(ranks)):
	rank_in += [(i,ranks[i])]

rank_in.sort(key=lambda x: x[1])

pr_rank, _ = zip(*rank_in)
pr_rank = list(pr_rank)
result2 = sm.OLS(y,x.iloc[:,pr_rank]).fit()
print(result2.summary())