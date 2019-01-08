<<<<<<< HEAD
import xlrd
from math import log
=======
import xlrd,pandas
>>>>>>> d3d94e2f339c772f1d1e8009b9bbc39210412d82

ti = xlrd.open_workbook("Titles.xlsx").sheet_by_index(0)
su = xlrd.open_workbook("Summaries.xlsx").sheet_by_index(0)
fa = xlrd.open_workbook("Factors.xlsx").sheet_by_index(0)

##########
#Some useful commands for xlrd
# print(ti.nsheets)
# print(ti.sheet_names())
# print(t11.row_values(0))
# print(t11.col_values(0))
# print(t11.cell(0,0))
# print(t11.cell(0,0).value)
# print(t11.row_slice(rowx=0,start_colx=0,end_colx=2))

#######################################
#Response is the same for titles and summaries
# tcont = ti.col_slice(colx=7,start_rowx=0,end_rowx=1001)
# scont = su.col_slice(colx=7,start_rowx=0,end_rowx=1001)

# for i in range(len(tcont)):
# 	tcont[i] = tcont[i].value
# 	scont[i] = scont[i].value

# dif = []
# for i in range(1001):
# 	if tcont[i] != scont[i]:
# 		dif += [i]

# print(len(dif))
#########################################
#counts are the same for titles and summaries
# tsc = ti.col_values(3)
# tnc = ti.col_values(4)
# tc = ti.col_values(5)

# sc = su.col_values(3)
# ssc = su.col_values(4)
# snc = su.col_values(5)

# dif = []
# for i in range(1001):
# 	d1 = tsc[i] != ssc[i]
# 	d2 = tc[i] != sc[i]
# 	d3 = tnc[i] != snc[i]
# 	if d1 and d2 and d3:
# 		dif += [i]

# print(len(dif))
##########################################

tsc = ti.col_values(colx=3,start_rowx=1,end_rowx=1001)
tnc = ti.col_values(colx=4,start_rowx=1,end_rowx=1001)
tc = ti.col_values(colx=2,start_rowx=1,end_rowx=1001)

sc = su.col_values(colx=3,start_rowx=1,end_rowx=1001)
ssc = su.col_values(colx=4,start_rowx=1,end_rowx=1001)
snc = su.col_values(colx=5,start_rowx=1,end_rowx=1001)
scont = su.col_slice(colx=7,start_rowx=1,end_rowx=1001)

for i in range(len(scont)):
	scont[i] = scont[i].value


print([sum(tc)/20000,sum(tsc)/20000,sum(tnc)/20000])
print([sum(sc)/20000,sum(ssc)/20000,sum(snc)/20000])


# tcont = []
# for i in range(1000):
# 	thr = [tc[i], tsc[i], tnc[i]]
# 	d = max(thr)
# 	if thr.count(d) > 1:
# 		ans = "unknown"
# 	else:
# 		ind = thr.index(d)
# 		ans = ["controversial","somewhat controversial","not controversial"][ind]

# 	tcont += [ans]

# dif = []
# for i in range(1000):
# 	if tcont[i] != scont[i] and tcont[i] != "unknown" and scont[i] != "unknown":
# 		dif += [i]

# for i in dif:
# 	print(tcont[i],scont[i])
	# thr = [tc[i], tsc[i], tnc[i]]
	# thp = [sc[i], ssc[i], snc[i]]
	# print(i+2,thr,thp)

# print(len(dif))
#########################################################
# scont = su.col_slice(colx=7,start_rowx=1,end_rowx=1001)

# coln = fa.row_values(0)

# ind = [85,50,83,94,49,33,6,61,35,7,2,8,71,63,32,10,70,37,29,4,72,58,45,64,25,42,31,51,67,59,68,20]
# se = [coln[i] for i in ind]

# print(se)