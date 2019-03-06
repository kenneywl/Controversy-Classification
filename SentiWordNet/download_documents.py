from googlesearch import search
from bs4 import BeautifulSoup
import requests
from difflib import SequenceMatcher
import pandas as pd
from interruptingcow import timeout

ti = pd.read_excel("Titles.xlsx")
su = pd.read_excel("Summaries.xlsx")

#filter to get the indexes of controversial not controversial
index = []
for i in range(1000):
	cont = su.loc[i,'Classification'] == 'controversial'
	ncont = su.loc[i,'Classification'] == 'not controversial'
	if cont or ncont:
		index += [i]

titles = pd.DataFrame(data={"Title": ti.loc[index,'Title'], "Response": su.loc[index,'Classification']},index=index)

###########################################################################

#This function takes a news title, searches google for it. takes num=3 off the top,
#then extracts only the words. This function outputs a dictionary {title,body}
#The first is the news title, the second is the body of the article as a single string.

def clean(line):
	line = line.replace('\n','')
	line = line.replace('\t','')
	line = line.strip()
	return(line)

def download_string(title,num=3):
	google_search = []
	for j in search(title, tld="com", num=num, stop=1, pause=3):
		google_search += [j]

	diffs = []
	for i in google_search:
		r = requests.get(i)
		soup = BeautifulSoup(r.text, 'html.parser')
		try:
			diffs += [SequenceMatcher(None, clean(title), clean(soup.title.string)).ratio()]
		except:
			diffs += [-1*r.status_code]

	title_closeness = max(diffs)
	title_closeness_ind = diffs.index(title_closeness)
	r = requests.get(google_search[title_closeness_ind])
	soup = BeautifulSoup(r.text, 'html.parser')

	strings = []
	for i in soup.find_all('p'):
		strings += [i.get_text()]


	title_new = clean(soup.title.string)
	body  = clean(" ".join(strings))

	strings = {"Title": title_new, "Title Requested": title, "Title_Closeness": title_closeness, "Body" : body}
	return(strings)


articles = pd.DataFrame()
for i in titles.index:
	tt = "\'" + clean(titles['Title'].loc[i]) + "\'"
	try:
		with timeout(12):
			st = download_string(tt)
			st['Response'] = titles['Response'].loc[i]
			print(i)
			print(st['Title'])
			print(tt,'\n')
	except:
			print(i,'Timeout:',tt,'\n')
			st = {"Title": "Timeout", "Title Requested": tt, "Title_Closeness": [-1], "Body" : None, 'Response': titles['Response'].loc[i]}

	st = pd.DataFrame(data=st,index=[i])
	articles = articles.append(st)


articles.to_pickle("articles.pkl")
print("Done")