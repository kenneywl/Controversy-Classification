import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold


#Lets import our data:
# summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
# titles = pd.read_pickle("titles.pkl")
# titles_r = pd.read_pickle("titles_r.pkl")

# nn = np.empty(shape=(1000,3))

# for i in range(1000):
# 	nn[i,] = [1,2,3]

# print(nn)

print(list(np.unique(summaries_r)))