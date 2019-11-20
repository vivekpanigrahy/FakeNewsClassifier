import pandas as pd

df_train = pd.read_csv("./data/train.csv")
df_train = df_train.drop(df_train.ix[:,'Unnamed: 6':'Unnamed: 8'].head(0).columns, axis=1)
print(df_train.head(10))

df_val = pd.read_csv("./data/validation.csv")
df_val = df_val.drop('Unnamed: 6', axis=1)
print(df_val.head(10))

df_test = pd.read_csv("./data/test.csv")
df_test = df_test.drop(df_test.ix[:,'Unnamed: 5':'Unnamed: 6'].head(0).columns, axis=1)
print(df_test.head(10))
