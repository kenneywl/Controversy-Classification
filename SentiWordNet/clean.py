import pandas as pd
import unicodedata
from nltk.corpus import stopwords
from textblob import Word
import string

def clean(x):
	def clean2(text):
		chars = string.punctuation + "0123456789"
		for i in chars:
			text = text.replace(i," ")
		return(text)

	x = str(x)
	x = x.lower()
	x = unicodedata.normalize("NFKD", x)
	x = clean2(x)
	x = x.replace("chinadaily", " ")
	x = x.replace("page", " ")

	li = x.split()
	li = [x for x in li if len(x) > 2]
	li = [x for x in li if not x in stopwords.words('english')]
	return(li)