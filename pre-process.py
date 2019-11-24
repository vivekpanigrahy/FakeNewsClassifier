import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher

stop_words = set(stopwords.words('english'))

df_train = pd.read_csv("./data/train.csv")
df_train = df_train.drop(df_train.ix[:,'Unnamed: 6':'Unnamed: 8'].head(0).columns, axis=1)
df_train = df_train.dropna(subset =['id'])
df_train = df_train.dropna(subset =['tid1'])
df_train = df_train.dropna(subset =['tid2'])
df_train = df_train.dropna(subset =['title1_en'])
df_train = df_train.dropna(subset =['title2_en'])
df_train = df_train.dropna(subset =['label'])
# print("Training set :")
# print(df_train.head(10))

# df_val = pd.read_csv("./data/validation.csv")
# df_val = df_val.drop('Unnamed: 6', axis=1)
# # print("Validation set :")
# # print(df_val.head(10))

# df_test = pd.read_csv("./data/test.csv")
# df_test = df_test.drop(df_test.ix[:,'Unnamed: 5':'Unnamed: 6'].head(0).columns, axis=1)
# # print("Test set :")
# # print(df_test.head(10))

temp_en = []
temp_title = []

for each_row in range(len(df_train)):
    temp_en.append(df_train.iloc[each_row][3])
    temp_title.append(df_train.iloc[each_row][4])
print(temp_en[0:10])
print(temp_title[0:10])            
temp =[]
temp1 =[]
for i in range(len(temp_en)):
    each_sentence=temp_en[i]
    each_sentence_title = temp_title[i]

    if len(each_sentence) >0:
        sen_l =each_sentence.split()
        sen_l_title = each_sentence_title.split()
        s=""
        r=""
        for word in sen_l:
            if word in stop_words:
                pass
            else:
                s+=word+" "
        temp_en[i]=s

        for word in sen_l_title:
            if word in stop_words:
                pass
            else:
                r+=word+" "
        temp_title[i] = r
    else:
        continue
# print(temp_en[0:10])
# print(temp_title[0:10])
