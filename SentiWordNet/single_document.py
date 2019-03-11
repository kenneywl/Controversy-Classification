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
				pos_score = sum(pos_score)/len(word_sysnets)
				neg_score = sum(neg_score)/len(word_sysnets)

				
				word_ave += [(pos_score,neg_score)]
	#Now the average for all words:
	try:
		pos_score, neg_score = zip(*word_ave)
		pos_score = sum(pos_score)/len(pos_score)
		neg_score = sum(neg_score)/len(neg_score)

		docu_ave = {"Word Count": len(text_t), "P/N Metric": pos_score/neg_score, "PN Metric": pos_score*neg_score, "P+N Metric": pos_score + neg_score, "P-N Metric": pos_score - neg_score, "Pos Score": pos_score,"Neg Score": neg_score}
	except:
		tt = "Div/Log Error"
		docu_ave = {"Word Count": len(text_t), "P/N Metric": tt, "PN Metric": tt, "P+N Metric": tt, "P-N Metric": tt, "Pos Score": tt,"Neg Score": tt}
	
	return(docu_ave)

# with open("test_ncont.txt") as file:
# 	ncont = file.read()

# with open("test_cont.txt") as file:
# 	cont = file.read()

# with open("test_file.txt") as file:
# 	file1 = file.read()

# with open("test_file_2.txt") as file:
# 	file2 = file.read()


# print("Company committed to providing electricity to rural areas")
# print(swn_single(ncont))
# print("Hometown Dilian")
# print(swn_single(file2))
# print("Company's sexist ads offensive and illegal")
# print(swn_single(cont))
# print("1000 people murdered by police")
# print(swn_single(file1))
