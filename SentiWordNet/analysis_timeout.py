from googlesearch import search
from bs4 import BeautifulSoup
import requests
from difflib import SequenceMatcher
import pandas as pd
from interruptingcow import timeout


pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000) 

ti = pd.read_excel("Titles.xlsx")
art = pd.read_pickle("articles_m.pkl")

errors = []
for i in art.index:
	if art.loc[i,'Word Count'] == "Timeout":
		errors += [i]

titles = ti.loc[errors,'Title']
resp = art.loc[errors,'Response']

time = pd.DataFrame(data={"Title":titles,"Response":resp},index=errors)

###########################################################################

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
for i in time.index:
	tt = "\'" + clean(time['Title'].loc[i]) + "\'"
	try:
		with timeout(12):
			st = download_string(tt)
			st['Response'] = time['Response'].loc[i]
			print(i)
			print(st['Title'])
			print(tt,'\n')
	except:
			print(i,'Timeout:',tt,'\n')
			st = {"Title": "Timeout", "Title Requested": tt, "Title_Closeness": [-1], "Body" : None, 'Response': time['Response'].loc[i]}

	st = pd.DataFrame(data=st,index=[i])
	articles = articles.append(st)

articles.to_pickle("timeout.pkl")