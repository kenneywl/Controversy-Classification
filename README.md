# Controversy-Reaserch

#This is my ongoing research on Controversy.

#The input files:

Factors.xlsx : LWIC data from the articles. 
Summaries.xlsx : Results from Mechinical Turk.
Titles.xlsx : Results from Mechinical Turk.

#The work files:

master.py : Calls all the other functions to get the results all in one place.
agreement.py : Looks at the agreement of from the results of the Mechanical Turk.  
agreement2.py : Looks at alternate ways to understand the info from Mech Turk.
cleandata.py : Cleans data. Adds Entropy and Standard Deviation as factors.
KNN3.py : Finds k for max accuracy and max weighed AUC. Calls functions from KNN_plot_func.py  
KNN_plot_func.py : Functions used for plotting KNN stuff.
NaiveBayes2.py : 10 fold cross and AUC graphs for Naive Bayes.
SVM.py : 10 fold cross for Support Vector Machines.
ChangeInEntropy.py : Calculation of KL divergence from summaries to titles.

#The results from cleandata.py above.

summaries.pkl : pickeled cleaned predictors for summaries.
summaries_r.pkl : pickled cleaned response for sumamries.
titles.pkl : pickled cleaned predictors for titles.
titles_r.pkl : pickled cleaned response for titles.

summaries_disp.pkl : Entropy of the summaries responses.
titles_disp.pkl : Entropy of the titles responses.

#Extranous files.

ROC.pdf : Paper on ROC and AUC.
LIWC2015.pdf : Paper on LWIC.
whatevs.py : I use this as a test file to help resolve programming issues.

#Folders.

Old_Scripts : A graveyard for scripts no longer needed. (ignore this)
pycache : Cache for some of the scripts above. (ignore this) 
