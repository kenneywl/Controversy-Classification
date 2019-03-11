from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import pandas as pd
import unicodedata
from nltk.corpus import stopwords
from textblob import Word
from gensim.models.doc2vec import TaggedDocument
import string
from gensim.utils import SaveLoad
from clean import clean
from math import log
from nltk.corpus import sentiwordnet as swn

art = pd.read_pickle("art.pkl")
su = pd.read_excel("Summaries.xlsx",nrows=1000)
idd = [i-1 for i in su['DB ID']]
art = art.loc[idd,:]
art.index = idd
su.index = idd

####################################################################################
#####################################################################################
#make dic
# labeled_sentances = []
# k = 1001
# for index,row in art.iterrows():
# 	k -= 1
# 	if k % 100 ==0:
# 		print(k)
# 	try:
# 		body = str(row['content'])+str(row['title'])
# 		cleaned = clean(body)
# 		labeled_sentances.append(cleaned)
# 	except:
# 		print('Error Index:',index)

# dct = Dictionary(labeled_sentances)  # fit dictionary

# dct.save("doc.dic")


####################################################################################
####################################################################################
####################################################################################


dct = SaveLoad.load("doc.dic")

def single_doc(doc_tokenized):

	#process a single word
	pos_score, neg_score = 0,0
	total_doc_freq = 0
	for word in doc_tokenized:
		word_synsets = list(swn.senti_synsets(word))
		pos_syn_score, neg_syn_score = 0,0
		if len(word_synsets) != 0:
			for syn in word_synsets:
				pos_syn_score += syn.pos_score()
				neg_syn_score += syn.neg_score()

			pos_syn_score /= len(word_synsets)
			neg_syn_score /= len(word_synsets)

			token_id = dct.token2id[word]
			unique_doc_frequency = 1000-dct.dfs[token_id]

			pos_syn_score *= unique_doc_frequency
			neg_syn_score *= unique_doc_frequency

		pos_score += pos_syn_score
		neg_score += neg_syn_score

	pos_score /= len(doc_tokenized)
	neg_score /= len(doc_tokenized)

	docu_ave = {"Word Count": len(doc_tokenized),"P+N Metric": pos_score + neg_score, "Abs(P-N) Metric": abs(pos_score - neg_score), "Pos Score": pos_score,"Neg Score": neg_score}

	return(docu_ave)

swn_metric = pd.DataFrame()
k = 1001
for i in art.index:
	title_body = clean(art.loc[i,'content'] + art.loc[i,'title'])
	data = single_doc(title_body)

	data['Response'] = su.loc[i,'controversial']-su.loc[i,'not controversial']
	data['Body'] = art.loc[i,'content']
	st = pd.DataFrame(data=data,index=[i])
	swn_metric = swn_metric.append(st)
	k -= 1
	if k % 100 == 0:
		print(k)

swn_metric.to_pickle("art_idf_m.pkl")