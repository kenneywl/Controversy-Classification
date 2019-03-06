import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.stats import dirichlet
import scipy.integrate as integrate

art = pd.read_pickle("articles_m.pkl")

#To get rid of the errors
no_error = []
for i in art.index:
	if not isinstance(art.loc[i,'P/N Metric'], str):
		no_error += [i]

art = art.loc[no_error]

#To whittle down to which titles are "real"
# ah = art['Title_Closeness'] > .01
# art = art[ah]

#separate in to cont and ncont
art_n = art[art.Response == 'not controversial']
art_c = art[art.Response == 'controversial']

nc = "Not Controversial n=" + str(art_n.shape[0])
cc = "Controversial n=" + str(art_c.shape[0])

######################################################################