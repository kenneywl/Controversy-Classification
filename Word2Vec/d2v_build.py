import pandas as pd
import string
from textblob import Word
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
import unicodedata
from random import shuffle

art = pd.read_pickle('art.pkl')
su = pd.read_excel("Summaries.xlsx",nrows=1000)

su.index = art.index


def resp(row):
	a = row['controversial']
	b = row['not controversial']
	ans = a-b
	return(ans)

art['response'] = su.apply(resp,axis=1)

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


labeled_sentances = []
for index,row in art.iterrows():
	try:
		cleaned = clean_lemma(row['content'])
		tags = [str(row['response']) + "_" + str(index)]
		labeled_sentances.append(TaggedDocument(words=cleaned,tags=tags))
	except:
		print('Error Index:',index)
		pass

model = Doc2Vec(min_count=2, window=6, vector_size=100, sample=1e-4, negative=5, workers=8)




model.build_vocab(labeled_sentances)

for epochs in range(20):
	shuffle(labeled_sentances)
	model.train(labeled_sentances,total_examples=model.corpus_count,epochs=model.epochs)
	print("Epoch:",epochs," Finished")

model.save("doument_model_continuous.d2v")