from KNN4 import *
from NaiveBayes2 import *
from sklearn.preprocessing import RobustScaler

from KNN_plot_func import *

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

#robust scaling causes every value (except NB) to increase, sometimes dramatically.
#The output as it comes out here is optimal, as far as I am concerned.
titles = pd.DataFrame(RobustScaler().fit_transform(titles))
summaries = pd.DataFrame(RobustScaler().fit_transform(summaries))

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
#  AUC max info: 50 0.6718954466717043 
#  ACC max info: 29 0.5599104143337066
# Naive Bayes:
#  AUC: 0.6806415034808792 
#  ACC: 0.5587905935050392
#
# For the Summaries:
# KNN:
#  K checked up to:  100 
#  AUC max info: 30 0.8088598393957359 
#  ACC max info: 33 0.675531914893617
# Naive Bayes:
#  AUC: 0.815540513880923 
#  ACC: 0.6510638297872341