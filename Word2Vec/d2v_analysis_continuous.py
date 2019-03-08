import pandas as pd
from gensim.models import Doc2Vec
import matplotlib.pyplot as plt
import statsmodels.api as sm

model = Doc2Vec.load("doument_model_continuous.d2v")

x = pd.DataFrame(columns=range(100))
y = pd.Series()
for i in model.docvecs.doctags:
	resp_split = i.split("_")
	index = int(resp_split[1])
	resp = resp_split[0]

	x.loc[index] = model[i]
	y.loc[index] = float(resp)


fit = sm.OLS(y,x).fit()

print(fit.summary())