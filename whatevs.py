import pandas as pd
from math import sqrt,log
from sklearn.preprocessing import RobustScaler

#For the factors:
fa = pd.read_excel("Factors.xlsx")
fa_ndm = pd.DataFrame(fa)

print(fa_ndm.iloc[:,2:])