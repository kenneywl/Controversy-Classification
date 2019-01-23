import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold


#Lets import our data:
# summaries = pd.read_pickle("summaries.pkl")
# summaries_r = pd.read_pickle("summaries_r.pkl")
# titles = pd.read_pickle("titles.pkl")
# titles_r = pd.read_pickle("titles_r.pkl")

def test(x,y):
	return(x,y)

a,b = test(1,2)

print(a,b)