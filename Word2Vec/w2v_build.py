import pandas as pd
import gensim as gs
import string
from textblob import Word
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import unicodedata
from gensim.test.utils import get_tmpfile

art = pd.read_pickle('art.pkl')
su = pd.read_excel("Summaries.xlsx",nrows=1000)

index = []
for i in range(1000):
	cont = su.loc[i,'Classification'] == 'controversial'
	ncont = su.loc[i,'Classification'] == 'not controversial'
	if cont or ncont:
		index += [i]

su = su.iloc[index]
art = art.iloc[index]

su.index = index
art.index = index

art['Response'] = su["Classification"]

def clean_lemma(x):
	def clean2(text):
		chars = string.punctuation + "0123456789"
		for i in chars:
			text = text.replace(i," ")
		return(text)

	x = str(x).lower()
	x = unicodedata.normalize("NFKD", x)
	x = clean2(x)
	x = x.replace("chinadaily", " ")
	x = x.replace("page", " ")

	li = x.split()
	li = [x for x in li if len(x) > 2]
	li = [x for x in li if not x in stopwords.words('english')]
	li = [Word(x).lemmatize() for x in li]
	return(li)

art['content'] = art['content'].apply(clean_lemma)

art_n = art[art.Response == 'not controversial']
art_c = art[art.Response == 'controversial']

word_list_n = art_n['content'].tolist()
word_list_c = art_c['content'].tolist()

model_n = Word2Vec(word_list_n,alpha=.05,min_count=2,window=5,size=300,workers=4)
model_c = Word2Vec(word_list_c,alpha=.05,min_count=2,window=5,size=300,workers=4)

model_n.save("model_n.bin")
model_c.save("model_c.bin")
