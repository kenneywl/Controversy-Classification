import pandas as pd
import matplotlib.pyplot as plt

articles_m = pd.read_pickle("articles_m.pkl")

articles_m.index = range(articles_m.shape[0])

#How balanced are they? Not very balanced. about .84 not controversial .16 controversial

print(list(articles_m))

cont = []
ncont = []
for i in range(articles_m.shape[0]):
	if articles_m.iloc[i,6] == "controversial":
		cont += [i]
	else:
		ncont += [i]

controversial = articles_m.iloc[cont]
ncontroversial = articles_m.iloc[ncont]

f1 = plt.figure()
plt.scatter(ncontroversial['Neg Score'],ncontroversial['Pos Score'],label="Not Controversial")
plt.scatter(controversial['Neg Score'],controversial['Pos Score'],label="Controversial")
plt.xlabel("Negative Score")
plt.ylabel("Positive Score")
plt.legend(loc="lower right")
plt.show()

