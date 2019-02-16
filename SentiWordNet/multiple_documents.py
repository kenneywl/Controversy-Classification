from single_document import swn_single
import pandas as pd

articles_c = pd.read_pickle("articles_c.pkl")

swn_metric = pd.DataFrame()
errors_ind = []
for i in range(articles_c.shape[0]):
	data = swn_single(articles_c.iloc[i,1])
	data['Response'] = articles_c.iloc[i,2]
	st = pd.DataFrame(data=data,index=[i])
	if st.iloc[0,2] != 0:
		swn_metric = swn_metric.append(st)
	else:
		errors_ind += [i]
	print(i)

swn_metric.to_pickle("articles_m.pkl")
print("Done. There were",len(errors_ind),"errors")
