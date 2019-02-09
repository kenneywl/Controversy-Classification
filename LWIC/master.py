from KNN4 import *
from NaiveBayes2 import *
from KNN_plot_func import *
import pandas as pd

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

agreements = pd.read_pickle("agreements.pkl")
agreements_r = pd.read_pickle("agreements_r.pkl")


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

print("For the agreements:")
print("KNN:")
knn_which_k(agreements,agreements_r)
knn_plot_auc(agreements,agreements_r,11)
print("Naive Bayes:")
nb_print_plot(agreements,agreements_r)

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

# For the agreements:
# KNN:
#  K checked up to:  100 
#  AUC max info: 10 0.7518674966075293 
#  ACC max info: 11 0.6203703703703703
# Naive Bayes:
#  AUC: 0.8386246055424439 
#  ACC: 0.6728395061728395

#From SVM.py we get:

# On linear Titles:
# {'C': 0.015} ACC: 0.5768660054053314
# On linear Summaries:
# {'C': 0.275} ACC: 0.6918303128535432
# On rbf Titles:
# {'C': 16, 'gamma': 0.0004} ACC: 0.5768537953931212
# On rbd Summaries:
# {'C': 18, 'gamma': 0.0007} ACC: 0.6841890015690055
# On sigmoid Titles:
# {'C': 0.9, 'gamma': 0.0008} ACC: 0.5768793129467287
# On sigmoid Summaries:
# {'C': 11, 'gamma': 0.0015} ACC: 0.6661243571762506