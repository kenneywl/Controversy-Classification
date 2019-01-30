from KNN4 import *
from NaiveBayes2 import *
from KNN_plot_func import *
import pandas as pd

summaries_ndm = pd.read_pickle("summaries_ndm.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles_ndm = pd.read_pickle("titles_ndm.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

agreements_ndm = pd.read_pickle("agreements_ndm.pkl")
agreements_r = pd.read_pickle("agreements_r.pkl")

print("For ndm titles:")
print("KNN:")
knn_which_k(titles_ndm,titles_r)
# knn_plot_auc(titles_ndm,titles_r,50)

print("Naive Bayes:")
nb_print_plot(titles_ndm,titles_r)

print("\nFor ndm Summaries:")
print("KNN:")
knn_which_k(summaries_ndm,summaries_r)
# knn_plot_auc(summaries_ndm,summaries_r,30)

print("Naive Bayes:")
nb_print_plot(summaries_ndm,summaries_r)

print("For ndm agreements:")
print("KNN:")
knn_which_k(agreements_ndm,agreements_r)
# knn_plot_auc(agreements_ndm,agreements_r,11)

print("Naive Bayes:")
nb_print_plot(agreements_ndm,agreements_r)


# For ndm titles:
# KNN:
#  K checked up to:  100 
#  AUC max info: 5 0.5862285405927361 
#  ACC max info: 42 0.48824188129899215
# Naive Bayes:
#  AUC: 0.6285454395490545 
#  ACC: 0.5050391937290034

# For ndm Summaries:
# KNN:
#  K checked up to:  100 
#  AUC max info: 11 0.6878822444557903 
#  ACC max info: 12 0.6563829787234042
# Naive Bayes:
#  AUC: 0.6959311374529229 
#  ACC: 0.5542553191489362
# For ndm agreements:
# KNN:
#  K checked up to:  100 
#  AUC max info: 5 0.7524398511912797 
#  ACC max info: 6 0.6203703703703703
# Naive Bayes:
#  AUC: 0.7378580611818344 
#  ACC: 0.595679012345679