import xlrd
from math import log,sqrt
import scipy.stats

ti = xlrd.open_workbook("Titles.xlsx").sheet_by_index(0)
su = xlrd.open_workbook("Summaries.xlsx").sheet_by_index(0)
fa = xlrd.open_workbook("Factors.xlsx").sheet_by_index(0)

tsc = ti.col_values(colx=3,start_rowx=1,end_rowx=1001)
tnc = ti.col_values(colx=4,start_rowx=1,end_rowx=1001)
tc = ti.col_values(colx=2,start_rowx=1,end_rowx=1001)

sc = su.col_values(colx=3,start_rowx=1,end_rowx=1001)
ssc = su.col_values(colx=4,start_rowx=1,end_rowx=1001)
snc = su.col_values(colx=5,start_rowx=1,end_rowx=1001)
scont = su.col_slice(colx=7,start_rowx=1,end_rowx=1001)

for i in range(len(scont)):
	scont[i] = scont[i].value

# print([sum(tc)/20000,sum(tsc)/20000,sum(tnc)/20000])
# print([sum(sc)/20000,sum(ssc)/20000,sum(snc)/20000])

#######################################################
#G-test on titles:
# def igzero(canidate):
# 	if canidate == 0:
# 		return 1
# 	else:
# 		return canidate

# etc = 20*sum(tc)/(20*1000)
# etsc = 20*sum(tsc)/(20*1000)
# etnc = 20*sum(tnc)/(20*1000)

# gts = 0
# for i in range(len(tc)):
# 	gts += tc[i]*log(igzero(tc[i])/etc)+ \
# 	tsc[i]*log(igzero(tsc[i])/etsc)+tnc[i]*log(igzero(tnc[i])/etnc)

# gts *= 2

# print(gts)
# print(1-scipy.stats.chi2.cdf(gts,999*2))
#g test st is 3703, pvalue about zero.
############################################################
# chi-square test on titles

# etc = 20*sum(tc)/(20*1000)
# etsc = 20*sum(tsc)/(20*1000)
# etnc = 20*sum(tnc)/(20*1000)

# chitst = 0
# for i in range(len(tc)):
# 	chitst += (tc[i]-etc)**2/etc+(tsc[i]-etsc)**2/etsc+(tnc[i]-etnc)**2/etnc

# print(chitst)
# print(1-scipy.stats.chi2.cdf(chitst,999*2))
#chisq test st is 3528, pvalue about zero.
#######################################################
#Fleiss's Kappa for titles:
# agt = []
# for i in range(len(tc)):
# 	agt += [(tc[i]**2+tsc[i]**2+tnc[i]**2-20)/380]

# po = (sum(agt)/1000)
# pe = (sum(tc)**2+sum(tsc)**2+sum(tnc)**2)/((20*1000)**2)

# kappa = (po-pe)/(1-pe)
# print("Kappa:", kappa)
#Kappa ts: .036 super low.

# pe3 = (sum(tc)**3+sum(tsc)**3+sum(tnc)**3)/((20*1000)**3)
# var = 2*(pe-(2*20-3)*pe**2+2*(20-2)*pe3)/(20*1000*(20-1)*(1-pe)**2)
# se = 1.96*sqrt(var)
# conf = [kappa-se,kappa+se]
# print("95% Conf Interval:", conf)

# Kappa: 0.03662505511683963
# 95% Conf Interval: [0.032896316987151876, 0.04035379324652739]

########################################################
# Fleis kappa for summaries:
# ags = []
# for i in range(len(sc)):
# 	ags += [(sc[i]**2+ssc[i]**2+snc[i]**2-20)/380]

# po = (sum(ags)/1000)
# pe = (sum(sc)**2+sum(ssc)**2+sum(snc)**2)/((20*1000)**2)

# kappa = (po-pe)/(1-pe)
# print("Kappa:", kappa)
# # Kappa ts: .139. better, but still low.
# # There is some agreement.
# #We can do a quick stat test.

# pe3 = (sum(sc)**3+sum(ssc)**3+sum(snc)**3)/((20*1000)**3)
# var = 2*(pe-(2*20-3)*pe**2+2*(20-2)*pe3)/(20*1000*(20-1)*(1-pe)**2)
# se = 1.96*sqrt(var)
# conf = [kappa-se,kappa+se]
# print("95% Conf Interval:", conf)

# Kappa: 0.13908545051989496
# 95% Conf Interval: [0.13259983939110698, 0.14557106164868294]

########################################################
########################################################
#lets join them together element wise and see:

# jc = [tc[i]+sc[i] for i in range(1000)]
# jsc = [tsc[i]+ssc[i] for i in range(1000)]
# jnc = [tnc[i]+snc[i] for i in range(1000)]

# #Fleis kappa for the joined info:
# ags = []
# for i in range(len(jc)):
# 	ags += [(jc[i]**2+jsc[i]**2+jnc[i]**2-40)/(40*39)]

# meanags = (sum(ags)/1000)
# sumofpss = (sum(jc)**2+sum(jsc)**2+sum(jnc)**2)/((40*1000)**2)

# print((meanags-sumofpss)/(1-sumofpss))
#Kappa ts: .066, looks like combining them doesn't
#help, it makes it worse. 