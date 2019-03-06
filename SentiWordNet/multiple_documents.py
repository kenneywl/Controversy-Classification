from single_document import swn_single
import pandas as pd

articles = pd.read_pickle("articles2.pkl")
su = pd.read_excel("Summaries.xlsx")

swn_metric = pd.DataFrame()
tt = 'Timeout'
for i in articles.index:
	if articles['Title'].loc[i] == 'Timeout':
		data = {"Word Count": tt, "P/N Metric":tt, "PN Metric": tt, "P+N Metric": tt, "P-N Metric": tt, "Pos Score":tt,"Neg Score": tt}
	else:
		title_body = articles.loc[i,'Body'] + articles.loc[i,'Title'] + su.loc[i,'Summary']
		data = swn_single(title_body)

	data['Response'] = articles.loc[i,'Response']
	data['Title_Closeness'] = articles.loc[i,'Title_Closeness']
	data['Title'] = articles.loc[i,'Title']
	data['Body'] = articles.loc[i,'Body']
	st = pd.DataFrame(data=data,index=[i])
	swn_metric = swn_metric.append(st)
	print(999-i)

swn_metric.to_pickle("art_m.pkl")