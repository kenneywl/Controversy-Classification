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
from nltk import pos_tag, word_tokenize

art = pd.read_pickle("art.pkl")
su = pd.read_excel("Summaries.xlsx",nrows=1000)

su.index = art.index
####################################################################################
#####################################################################################
#make dic
# labeled_sentances = []
# k = 1001
# for i in art.index:
# 	k -= 1
# 	if k % 100 ==0:
# 		print(k)
# 	try:
# 		body = str(art.loc[i,'content'])+ " " +str(art.loc[i,'title'])
# 		cleaned = clean(body)
# 		cleaned = word_tokenize(cleaned)
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
	tags = pos_tag(doc_tokenized)

	pos_score, neg_score = 0,0
	for t in tags:
		word = t[0]
		pos = t[1][0]

		if pos == 'J': 
			part = 'a'
		elif pos == 'N': 
			part = 'n'
		elif pos == 'R':
			part = 'r'
		elif pos == 'V':
			part = 'v'
		else:
			part = "average"

		if part == "average":
			word_synsets = list(swn.senti_synsets(word))
		else:
			word_synsets = swn.senti_synsets(word,part)

		pos_syn_score, neg_syn_score = 0, 0
		if word_synsets == True: #an empty list is false!
			for syn in word_synsets:
				print(word_synsets)
				pos_syn_score += syn.pos_score()
				neg_syn_score += syn.neg_score()

				pos_syn_score /= len(word_synsets)
				neg_syn_score /= len(word_synsets)

		token_id = dct.token2id[word]
		unique_doc_freq = 1000-dct.dfs[token_id]

		pos_syn_score *= unique_doc_freq
		neg_syn_score *= unique_doc_freq

		pos_score += pos_syn_score
		neg_score += neg_syn_score

	pos_score /= len(doc_tokenized)
	neg_score /= len(doc_tokenized)

	docu_ave = {"Word Count": len(doc_tokenized),"P+N Metric": pos_score + neg_score, "Abs(P-N) Metric": abs(pos_score - neg_score), \
	             "Pos Score": pos_score,"Neg Score": neg_score}

	return(docu_ave)

# swn_metric = pd.DataFrame()
# k = 1001
# for i in art.index:
# 	title_body = clean(str(art.loc[i,'title']) + " " + str(art.loc[i,'content']))
# 	title_body = word_tokenize(title_body)
# 	try:
# 		data = single_doc(title_body)
# 	except Exception as error:
# 		print("Error:", error,"\n","Error ind:",i)

# 	data['Response'] = su.loc[i,'controversial']-su.loc[i,'not controversial']
# 	data['Body'] = art.loc[i,'content']
# 	st = pd.DataFrame(data=data,index=[i])
# 	swn_metric = swn_metric.append(st)
# 	k -= 1
# 	if k % 100 == 0:
# 		print(k)

# swn_metric.to_pickle("art_idf_m.pkl")

title_body = clean(str(art.loc[53,'title']) + " " + str(art.loc[53,'content']))
title_body = word_tokenize(title_body)
print(single_doc(title_body))