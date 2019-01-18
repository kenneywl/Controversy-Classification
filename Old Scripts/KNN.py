import pandas as pd
from sklearn.model_selection import train_test_split

#For KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#For plotting
import matplotlib.pyplot as plt

#####################################################################
#we unpickle our data from cleandata.py

summaries = pd.read_pickle("summaries.pkl")
summaries_r = pd.read_pickle("summaries_r.pkl")
titles = pd.read_pickle("titles.pkl")
titles_r = pd.read_pickle("titles_r.pkl")

######################################################################
#I wrote a function to do a 10 fold cross validation.
#It leaves out 10% at a time and tests against that.
#This function finds the best value for k by accuracy

def knn_which_k(factors,response,max_k=30):
	hold = int(factors.shape[0]/10)

	acc_m = []
	for k in range(1,max_k+1):
		knn = KNeighborsClassifier(n_neighbors = k)
		acc = []
		for i in range(10):
			ind = [j for j in range(hold*i,hold*(i+1))]

			set_p = factors.drop(ind)
			set_r = response.drop(ind)

			knn.fit(set_p,set_r)
			ypred = knn.predict(factors.iloc[ind,:])

			acc += [metrics.accuracy_score(response.iloc[ind],ypred)]
		acc_m += [sum(acc)/10]
	return(acc_m)

################################################################
#The accuracies for each value of k from 1 to 30
#for titles:

print("For the titles:")
acc_t = knn_which_k(titles,titles_r)

max_tk = acc_t.index(max(acc_t))
print("k:",max_tk+1,"\nmax acc:",max(acc_t))

#The k value is 21 with an acc of .494
#somewhat better than random (which would be .33)
#as there are three possible response categaores.

#For the summaries:
print("For the summaries:")
acc_s = knn_which_k(summaries,summaries_r)

max_sk = acc_s.index(max(acc_s))
print("k:",max_sk+1,"\nmax acc:",max(acc_s))

#The k value is 12, with a acc of .65
#not terrible.

#Lets graph it.

ks = [i+1 for i in range(30)]

f1 = plt.figure()
plt.plot(ks,acc_t)

plt.xlabel("K Nearest Neighbors")
plt.ylabel("Accuracy")
plt.title("Title Accuracy of K Nearest Neighbor")
f1.show()

f2 = plt.figure()
plt.plot(ks,acc_s)

plt.xlabel("K Nearest Neighbors")
plt.ylabel("Accuracy")
plt.title("Summaries Accuracy of K Nearest Neighbor")
f2.show()