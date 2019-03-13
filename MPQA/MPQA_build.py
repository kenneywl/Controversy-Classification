import pandas as pd


pd.options.display.float_format = '{:20,.3f}'.format
pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)

mp = pd.read_pickle("mpqa_features.pickle")

print(mp.head())