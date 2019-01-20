from KNN3 import *
from NaiveBayes3 import *

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

print("For the titles:")
print("KNN:")
which_k(titles,titles_r)
which_k_plot(titles,titles_r)
knn_plot_auc(titles,titles_r,10)

print("Naive Bayes:")
NB_print_AUC(titles,titles_r)
NB_plot_auc(titles,titles_r)

print("For the Summaries:")
print("KNN:")
which_k(summaries,summaries_r)
which_k_plot(summaries,summaries_r)
knn_plot_auc(summaries,summaries_r,14)

print("Naive Bayes:")
NB_print_AUC(summaries,summaries_r)
NB_plot_auc(summaries,summaries_r)