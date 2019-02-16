from googlesearch import search
from bs4 import BeautifulSoup
import requests
from difflib import SequenceMatcher
import pandas as pd
from interruptingcow import timeout

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
titles = pd.DataFrame(data={"Titles":ti.iloc[index,1],"Response":responses})

###########################################################################

#This function takes a news title, searches google for it. takes num=3 off the top,
#then extracts only the words. This function outputs a dictionary {title,body}
#The first is the news title, the second is the body of the article as a single string.

def download_string(title,num=3):
	google_search = []
	for j in search(title, tld="com", num=num, stop=1, pause=2):
		google_search += [j]

	diffs = []
	for i in google_search:
		r = requests.get(i)
		soup = BeautifulSoup(r.text, 'html.parser')
		diffs += [SequenceMatcher(None, title, soup.title.string).ratio()]

	r = requests.get(google_search[diffs.index(max(diffs))])
	soup = BeautifulSoup(r.text, 'html.parser')

	strings = []
	for i in soup.find_all('p'):
		strings += [i.get_text()]


	def clean(line):
		line = line.replace('\n','')
		line = line.replace('\t','')
		line = line.strip()
		return(line)

	title = clean(soup.title.string)
	body  = clean(" ".join(strings))

	strings = {"Titles": title, "Body" : body}
	return(strings)


articles = pd.DataFrame()
error_ind = []
for i in range(len(titles['Titles'])):
	try:
		with timeout(12):
			tt = titles['Titles'].iloc[i]
			st = download_string(tt)

			st = pd.DataFrame(data=st,index=[i])
			articles = articles.append(st)
	except:
		error_ind += [i]
	print(i)

error_ind = pd.Series(error_ind)
articles.index = range(articles.shape[0])
articles.to_pickle("articles.pkl")
error_ind.to_pickle("error_ind.pkl")
print("Done")