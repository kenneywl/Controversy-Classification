from KNN4 import *
from NaiveBayes2 import *
from KNN_plot_func import *
import pandas as pd

summaries = pd.read_pickle("summaries.pkl")
summaries_res = pd.read_pickle("summaries_res.pkl")
titles = pd.read_pickle("titles.pkl")
titles_res = pd.read_pickle("titles_res.pkl")

agreements = pd.read_pickle("agreements.pkl")
agreements_res = pd.read_pickle("agreements_res.pkl")

print("Naive Bayes:")
nb_print_plot(titles,titles_res)

print("\nFor the Summaries:")
print("KNN:")
knn_which_k(summaries,summaries_res)
knn_plot_auc(summaries,summaries_res,30)

print("Naive Bayes:")
nb_print_plot(summaries,summaries_res)

print("For the agreements:")
print("KNN:")
knn_which_k(agreements,agreements_res)
knn_plot_auc(agreements,agreements_res,11)
print("Naive Bayes:")
nb_print_plot(agreements,agreements_res)