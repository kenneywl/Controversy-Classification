# Controversy-Reaserch

#This is my ongoing research on Controversy.

#The input files:

1.  Factors.xlsx : LWIC data from the articles. 
2.  Summaries.xlsx : Results from Mechinical Turk.
3.  Titles.xlsx : Results from Mechinical Turk.

#The work files:

4.  agreement.py : Looks at the agreement of from the results of the Mechanical Turk.  
4a. agreement2.py : Looks at alternate ways to understand the info from Mech Turk.
5.  cleandata.py : Cleans data. Adds Entropy and Standard Deviation as factors.
6.  KNN3.py : Finds k for max accuracy and max weighed AUC. Calls functions from KNN_plot_func.py  
6a.  KNN_plot_func.py : Functions used for plotting KNN stuff.
8.  NaiveBayes2.py : 10 fold cross and AUC graphs for Naive Bayes.

#The results from cleandata.py above.

9.   summaries.pkl : pickeled cleaned predictors for summaries.
10.  summaries_r.pkl : pickled cleaned response for sumamries.
11.  titles.pkl : pickled cleaned predictors for titles.
12.  titles_r.pkl : pickled cleaned response for titles.

#Extranous files.

13.  ROC.pdf : Paper on ROC and AUC.
14.  LIWC2015.pdf : Paper on LWIC.
15.  whatevs.py : I use this as a test file to help resolve programming issues.

#Folders.

16.  Old_Scripts : A graveyard for scripts no longer needed. (ignore this)
17.  __pycache__ : Cache for some of the scripts above. (ignore this) 
