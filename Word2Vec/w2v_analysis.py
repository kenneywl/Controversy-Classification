import pandas as pd
from gensim.models import Word2Vec


model_n = Word2Vec.load('model_n.bin')
model_c = Word2Vec.load('model_c.bin')

# similar_n = model_n.wv.similar_by_word('government',topn=20)
# similar_c = model_c.wv.similar_by_word('government',topn=20)
# print(similar_n,"\n")
# print(similar_c)

# ss = model_n.wv.similarity("government","good")
# ss1 = model_c.wv.similarity("government","good")
# print(ss,ss1)

mn= model_n.wv
mc = model_c.wv

# for i in dd:
# 	print(i,":",dd[i].count)

# print(mn.similar_by_word("good"))
# print(mc.similar_by_word("good"))

print(mn.most_similar("good"))