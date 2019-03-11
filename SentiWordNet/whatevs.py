from gensim.corpora import Dictionary

dataset = ['driving car ',
           'drive car carefully',
           'student and university']

dataset = [d.split() for d in dataset]
print(dataset)

vocab = Dictionary(dataset)
print(vocab)