import pandas as pd
from nltk.corpus import sentiwordnet as swn, stopwords
from nltk.tokenize import RegexpTokenizer
from math import log


#The following takes a single string and processes it:
#output is a dictionary {metric, relative entropy, pos_score+neg_score, |pos_score - neg_score|, pos_score, neg_score}
def swn_single(document):
	stop_words = stopwords.words('english')
	tokenizer = RegexpTokenizer(r'\w+')
	text_t = tokenizer.tokenize(document)

	#lets remove stopwords:
	text_t = [i for i in text_t if not i in stop_words]
	word_ave = []
	for word in text_t:
			word_sysnets = list(swn.senti_synsets(word))
			score = [(i.pos_score(),i.neg_score()) for i in word_sysnets]
			#The following makes sure that sentwordnet gives us something!
			if 0 < len(score):
				pos_score, neg_score = zip(*score)
				pos_score = pos_score[0]
				neg_score = neg_score[0]
				# pos_score = sum(pos_score)/len(word_sysnets)
				# neg_score = sum(neg_score)/len(word_sysnets)
				word_ave += [(pos_score,neg_score)]
	#Now the average for all words:
	try:
		pos_score, neg_score = zip(*word_ave)
		pos_score = sum(pos_score)/len(pos_score)
		neg_score = sum(neg_score)/len(neg_score)

		rel_scores = (pos_score/(pos_score+neg_score),neg_score/(pos_score+neg_score))
		entropy = -(rel_scores[0]*log(rel_scores[0])+rel_scores[1]*log(rel_scores[1]))/log(2)

		sum_of = pos_score+neg_score
		absol_diff = abs(pos_score-neg_score)
	except:
		entropy, sum_of, absol_diff, pos_score, neg_score = (0,0,0,0,0)

	docu_ave = {"Metric": pos_score*neg_score, "Relative Entropy": entropy,
	"Sum of Scores": sum_of, "Absolute Difference": absol_diff, "Pos Score": pos_score,"Neg Score": neg_score}
	return(docu_ave)