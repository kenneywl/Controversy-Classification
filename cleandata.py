import pandas as pd

#For the factors:
fa = pd.read_excel("Factors.xlsx")
fa = pd.DataFrame(fa)

#######################################################################
#For the titles we don't have the result of majority voting.
#Note I used iss for "controvesial", som for "somewhat" and not for
#"not controversial". It makes the resp have the same length. Here we go:

ti = pd.read_excel("Titles.xlsx")
tsc = ti.iloc[0:1000,3]
tnc = ti.iloc[0:1000,4]
tc = ti.iloc[0:1000,2]

tcont = []
for i in range(1000):
	thr = [tc[i], tsc[i], tnc[i]]
	d = max(thr)
	if thr.count(d) > 1:
		ans = "unk"
	else:
		ind = thr.index(d)
		ans = ["iss","som","not"][ind]

	tcont += [ans]

titles_r = pd.DataFrame(tcont)[0]

#columns the paper says were used for the titles:
ind = [85,50,83,94,49,33,6,61,35,7,2,8,71,63,32,10,70,37,29,4,72, \
       58,45,64,25,42,31,51,67,59,68,20]

titles = fa.iloc[:,ind]

#"titles" should be the factors used for each model. titles_r should
#be the responses.
###################################################################
#For the summaries:
su = pd.read_excel("Summaries.xlsx")

scont = su.iloc[0:1000,7]

#make the repsonses the same length for ease.
#we can always put it back later.
for i in range(len(scont)):
	if scont[i] == "controversial":
		scont[i] = "iss"
	if scont[i] == "somewhat controversial":
		scont[i] = "som"
	if scont[i] == "not controversial":
		scont[i] = "not"
	if scont[i] == "unknown":
		scont[i] = "unk"


#columns the paper says were used for the summaries:

fa_fe = [i.lower() for i in list(fa)]
fa_ch = ["power","negemo","anger","tone","drives","percept","comma","see","discrep",\
		 "auxverb","dic","allpunc","risk","otherp","anx","function","pronoun","ppron",\
		  "cogproc","number","clout","negate","social","leisure","we","analytic",\
		  "differ","affect","achieve","certain","sad","tentat","work","prep","wc",\
		  "reward","wps","space","article","verb","hear","period","affiliation","relativ"]

ind2 = []
for i in fa_ch:
	for j in range(len(fa_fe)):
		if i == fa_fe[j]:
			ind2 += [j]

fach_su = fa.iloc[:,ind2]

#Note we are missing standard devation and entropy.
#Lets include them: (this might be wrong.)
from math import sqrt,log

sc = su.iloc[0:1000,3]
ssc = su.iloc[0:1000,4]
snc = su.iloc[0:1000,5]

#for stand dev
ssd = []
mean = 20/3
for i in range(len(sc)):
	ssd += [sqrt(((sc[i]-mean)**2+(ssc[i]-mean)**2+(snc[i]-mean)**2)/2)]

#for entropy
sen = []
for i in range(len(sc)):
	if sc[i] == 0: sc[i] = 1
	if ssc[i] == 0: ssc[i] = 1
	if snc[i] == 0: snc[i] = 1

	sen += [log(20)-(sc[i]*log(sc[i])+ssc[i]*log(ssc[i])+snc[i]*log(snc[i]))/20]

summaries = fach_su.assign(StandardDevaition = ssd,Entropy=sen)
summaries_r = scont
#There we go. Summaries are the factors, summaries_r is responses.

def info():
	print("Factors for titles: 'titles' \nResponses for the titles:'titles_r")
	print("Factors for summaries: 'summaries' \nResponses for the summaries:'summaries_r'")
