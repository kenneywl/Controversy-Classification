import pandas as pd
from sklearn.model_selection import train_test_split as tts

#For KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#####################################################################
#we can unpickle our data from cleandata.py

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

################################################################

x_train , x_test, y_train, y_test = tts(titles,titles_r,test_size=.1,shuffle=False)

print(x_train)
print(titles_r.shape[0])