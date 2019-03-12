
from nltk.corpus import sentiwordnet as swn

k = swn.senti_synsets('try')
print(len(k))

for i in k:
	print(i.pos_score())
