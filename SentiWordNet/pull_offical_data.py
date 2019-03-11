import pandas as pd

# art = pd.read_csv("articles_official.csv",usecols=[0,4,5,6,7,10,15]).iloc[:7844]

# art.to_pickle("art_off.pkl")

art = pd.read_pickle("art_off.pkl")
art_id = pd.read_excel("Summaries.xlsx",usecols=[0,1],nrows=1000)
rows = [i-1 for i in art_id['DB ID']]

art.loc[rows,:].to_pickle("art.pkl")



# from single_document import swn_single

# articles = art_all

# swn_metric = pd.DataFrame()
# k = 1000
# for i in articles.index:
# 	title_body = str(articles.loc[i,'content']) + str(articles.loc[i,'title'])
# 	data = swn_single(title_body)

# 	data['Response Cat'] = su.loc[i,'Classification']
# 	data['Response Num'] = su.loc[i,'controversial']-su.loc[i,'not controversial']
# 	data['Title'] = articles.loc[i,'title']
# 	data['Body'] = articles.loc[i,'content']
# 	data['Summary'] = articles.loc[i,'summary']
# 	st = pd.DataFrame(data=data,index=[i])
# 	swn_metric = swn_metric.append(st)
# 	k -= 1
# 	print(k)

# swn_metric.to_pickle("art_all_m.pkl")
# print("Done")