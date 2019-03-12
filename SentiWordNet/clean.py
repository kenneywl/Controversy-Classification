import pandas as pd
import unicodedata
from nltk.corpus import stopwords
from textblob import Word
import string

def clean(x):
	x = str(x)
	x = unicodedata.normalize("NFKD", x)
	x = x.lower()
	chars = string.punctuation + "0123456789"
	for i in chars:
		x = x.replace(i," ")
	return(x)