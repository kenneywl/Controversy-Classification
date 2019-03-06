import pandas as pd

time2 = pd.read_pickle("timeout.pkl")
ti = pd.read_excel("Titles.xlsx")
su = pd.read_excel("Summaries.xlsx")

no_errors = []
for i in time2.index:
	if time2.loc[i,'Title'] != "Timeout":
		no_errors += [i]

articles = pd.read_pickle("articles.pkl")

articles.drop(no_errors,inplace=True)
articles = articles.append(time2.loc[no_errors])

articles.to_pickle("articles2.pkl")


###########################################################################