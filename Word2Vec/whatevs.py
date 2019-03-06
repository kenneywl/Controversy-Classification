import pandas as pd
import gensim as gs
import string
from textblob import Word
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import unicodedata
stop = stopwords.words('english')

def clean_lemma(x):
	x = str(x)
	x = unicodedata.normalize("NFKD", x)
	x = x.replace('[^\w\s]',' ')
	x = x.replace('.',' ')
	x = x.replace('``', ' ')
	x = x.replace("''", ' ')
	x = x.replace('/',' ')
	x = x.replace('-',' ')
	x = x.replace("'", ' ')
	x = x.replace("u", ' ')
	x = x.replace("page", " ")
	x = x.replace("xa0", ' ')
	li = x.split()
	li = [x for x in li if len(x) > 2]
	li = [x for x in li if x not in string.punctuation]
	li = [x for x in li if not x.isdigit()]
	li = [x for x in li if not x in stop]
	li = [Word(x).lemmatize() for x in li]
	return(li)

kk =  string.punctuation + "this is a test of the emergency broadcastin" + "g station"

print(string.punctuation + "0123456789")

