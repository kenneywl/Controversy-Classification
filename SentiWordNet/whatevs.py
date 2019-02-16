import pandas as pd

articles = pd.read_pickle("articles.pkl")
error_ind = pd.read_pickle("error_ind.pkl")

ti = pd.read_excel("Titles.xlsx")
su = pd.read_excel("Summaries.xlsx")

classif = su['Classification'].iloc[:1000]

#filter to get the indexes of controversial not controversial
index = []
for i in range(classif.shape[0]):
	cont = classif[i] == 'controversial'
	ncont = classif[i] == 'not controversial'
	if cont or ncont:
		index += [i]

responses = [classif[i] for i in index]
responses = [responses[i] for i in range(len(responses)) if not i in error_ind]

articles['Responses'] = responses

print(articles.iloc[16,0])