from KNN4 import *
from NaiveBayes2 import *
from KNN_plot_func import *
import pandas as pd

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

print("For the titles:")
print("KNN:")
knn_which_k(titles,titles_r)
knn_plot_auc(titles,titles_r,50)

print("Naive Bayes:")
nb_print_plot(titles,titles_r)

print("\nFor the Summaries:")
print("KNN:")
knn_which_k(summaries,summaries_r)
knn_plot_auc(summaries,summaries_r,30)

print("Naive Bayes:")
nb_print_plot(summaries,summaries_r)

#This is what the paper did:
#Without robust scaling, k=30.

# For the titles:
# KNN:
#  K checked up to:  30 
#  AUC max info: 30 0.6537846111937079 
#  ACC max info: 29 0.5599104143337066
# Naive Bayes:
#  AUC: 0.6806415034808792 
#  ACC: 0.5587905935050392
#
# For the Summaries:
# KNN:
#  K checked up to:  30 
#  AUC max info: 30 0.8088598393957359 
#  ACC max info: 24 0.6723404255319149
# Naive Bayes:
#  AUC: 0.815540513880923 
#  ACC: 0.6510638297872341

###############################################################
#And lets see what we can do:

#with robust scaling, k=100 (maximum we can do)


# For the titles:
# KNN:
#  K checked up to:  100 
#  AUC max info: 78 0.662918003298281 
#  ACC max info: 15 0.5520716685330347
# Naive Bayes:
#  AUC: 0.6854088733755024 
#  ACC: 0.561030235162374

# For the Summaries:
# KNN:
#  K checked up to:  100 
#  AUC max info: 22 0.7877023370430536 
#  ACC max info: 33 0.6734042553191489
# Naive Bayes:
#  AUC: 0.7641053116164848 
#  ACC: 0.6202127659574468

#From SVM.py I need to redo due to a change in summaries.