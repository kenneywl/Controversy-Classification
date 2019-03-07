import pandas as pd

# art = pd.read_csv("articles_official.csv",usecols=[0,4,5,6,7,10,15]).iloc[:7844]

# art.to_pickle("art_off.pkl")

art = pd.read_pickle("art_off.pkl")
# art_id = pd.read_excel("Titles.xlsx",usecols=[0,1])
# rows = [i-1 for i in art_id['Art_Id']]

# art.loc[rows].to_pickle("art.pkl")

su = pd.read_excel("Summaries.xlsx",nrows=1000)
idd = [i-1 for i in su['DB ID']]
su.index = idd
art_all = art.loc[idd,:]

# print(sum(su["Classification"].value_counts()))

# index = []
# for i in range(1000):
# 	cont = su.loc[i,'Classification'] == 'controversial'
# 	ncont = su.loc[i,'Classification'] == 'not controversial'
# 	if cont or ncont:
# 		index += [i]

# su = su.iloc[index]
# su["Classification"].value_counts()


# art = art.iloc[index]

# su.index = index
# art.index = index

# art['Response'] = su["Classification"]

from single_document import swn_single

articles = art_all

swn_metric = pd.DataFrame()
for i in articles.index:
	title_body = str(articles.loc[i,'content']) + str(articles.loc[i,'title'])
	data = swn_single(title_body)

	# data['Response'] = articles.loc[i,'Response']
	data['Response'] = su.loc[i,'controversial']-su.loc[i,'not controversial']
	data['Title'] = articles.loc[i,'title']
	data['Body'] = articles.loc[i,'content']
	data['Summary'] = articles.loc[i,'summary']
	st = pd.DataFrame(data=data,index=[i])
	swn_metric = swn_metric.append(st)

swn_metric.to_pickle("art_all_m.pkl")