import pandas as pd
import string
from textblob import Word
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
import unicodedata
from gensim.test.utils import get_tmpfile
from random import shuffle

art = pd.read_pickle('art.pkl')
su = pd.read_excel("Summaries.xlsx",nrows=1000)

index = []
for i in range(1000):
	cont = su.loc[i,'Classification'] == 'controversial'
	ncont = su.loc[i,'Classification'] == 'not controversial'
	if cont or ncont:
		index += [i]

print(len(index))

'''
su = su.iloc[index]
art = art.iloc[index]

su.index = index
art.index = index

art['response'] = su["Classification"]

def clean_lemma(x):
	def clean2(text):
		chars = string.punctuation + "0123456789"
		for i in chars:
			text = text.replace(i," ")
		return(text)

	x = x.lower()
	x = unicodedata.normalize("NFKD", x)
	x = clean2(x)
	x = x.replace("chinadaily", " ")
	x = x.replace("page", " ")

	li = x.split()
	li = [x for x in li if len(x) > 2]
	li = [x for x in li if not x in stopwords.words('english')]
	li = [Word(x).lemmatize() for x in li]
	return(li)

print(art.index)

labeled_sentances = []
for index,row in art.iterrows():
	try:
		cleaned = clean_lemma(row['content'])
		tags = [row['response'] + "_" + str(index)]
		labeled_sentances.append(TaggedDocument(words=cleaned,tags=tags))
	except:
		pass

model = Doc2Vec(min_count=2, window=6, vector_size=100, sample=1e-4, negative=5, workers=8)

model.build_vocab(labeled_sentances)

for epochs in range(10):
	shuffle(labeled_sentances)
	model.train(labeled_sentances,total_examples=model.corpus_count,epochs=model.epochs)
	print("Epoch:",epochs," Finished")

model.save("doument_model.d2v")
'''