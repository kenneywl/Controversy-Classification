import pandas as pd
from math import log

# summaries_ent = pd.read_pickle("summaries_disp.pkl")
# titles_ent = pd.read_pickle("titles_disp.pkl")

ti = pd.read_excel("Titles.xlsx")
su = pd.read_excel("Summaries.xlsx")

titles_a = pd.DataFrame(ti.iloc[:,[2,3,4]])
summaries_a = pd.DataFrame(su.iloc[:1000,[3,4,5]])

summaries_a.iloc[:,2] = [float(summaries_a.iloc[i,2]) for i in range(1000)]
summaries_a = summaries_a.rename( \
	columns={"controversial":"Controversial","not controversial":"Not Controversial","somewhat controversial":"Somewhat Controversial"})

#we now look at the KL divergence; by using the repsonse of titles to approximate the summaries.

kl = []
for i in range(1000):
	real = summaries_a.iloc[i,]
	approx = titles_a.iloc[i,]

	inter = 0
	for j in range(3):
		a = real[j] == 0
		b = approx[j] == 0
		if a or b:
			real[j] = 1
			approx[j] = 1

		inter += real[j]*log(real[j]/approx[j])/log(3)

	kl += [inter]

print(sum(kl)/1000)