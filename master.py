from KNN3 import *
from NaiveBayes3 import *
from sklearn.preprocessing import RobustScaler

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")


# titles = pd.DataFrame(RobustScaler().fit_transform(titles))
# summaries = pd.DataFrame(RobustScaler().fit_transform(summaries))

print("For the titles:")
print("KNN:")
which_k(titles,titles_r,50)
#which_k_plot(titles,titles_r)
#knn_plot_auc(titles,titles_r,10)

print("Naive Bayes:")
NB_print_AUC(titles,titles_r)
#NB_plot_auc(titles,titles_r)

print("For the Summaries:")
print("KNN:")
which_k(summaries,summaries_r,50)
#which_k_plot(summaries,summaries_r)
#knn_plot_auc(summaries,summaries_r,14)

print("Naive Bayes:")
NB_print_AUC(summaries,summaries_r)
#NB_plot_auc(summaries,summaries_r)

#This is what the paper did:
#No robust scaling, k=30.

# For the titles:
# KNN:
#  AUC max info: 10 0.6036526659216251 
#  Accuracy max info: 20 0.4917727840199751
# Naive Bayes:
#  AUC: 0.6806788280248883
# For the Summaries:
# KNN:
#  AUC max info: 14 0.6857727840339154 
#  Accuracy max info: 12 0.65
# Naive Bayes:
#  AUC: 0.8155169741299784

###############################################################
#And lets see if we can better it:

#With robust scaling and k=30:
# For the titles:
# KNN:
#  AUC max info: 30 0.6537846111937079 
#  Accuracy max info: 29 0.5600998751560549
# Naive Bayes:
#  AUC: 0.6806415034808792
# For the Summaries:
# KNN:
#  AUC max info: 29 0.8081641462947167 
#  Accuracy max info: 24 0.672340425531915
# Naive Bayes:
#  AUC: 0.815540513880923


#With robust scaling k=50:
# For the titles:
# KNN:
#  AUC max info: 50 0.6718954466717043 
#  Accuracy max info: 29 0.5600998751560549
# Naive Bayes:
#  AUC: 0.6806415034808792
# For the Summaries:
# KNN:
#  AUC max info: 30 0.8088598393957359 
#  Accuracy max info: 33 0.675531914893617
# Naive Bayes:
#  AUC: 0.815540513880923