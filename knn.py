import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
import csv
from collections import Counter
import math
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
import time

start = time.process_time()

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)


df_train = pd.read_csv("./data/train_final.csv")
df_test = pd.read_csv("./data/test_mod.csv")

print("Data load complete in ", time.process_time() - start)

fake_title_train = []
title_train = []
label_train = []
for each_row in range(len(df_train)):
    fake_title_train.append(df_train.iloc[each_row][3])
    title_train.append(df_train.iloc[each_row][4])
    if df_train.iloc[each_row][5] == 'unrelated':
        label_train.append(0)
        continue
    elif df_train.iloc[each_row][5] == 'agreed':
        label_train.append(1)
        continue
    else:
        label_train.append(2)
print("Training Lists created")

similarity_train = []

for each_element in range(len(fake_title_train)):
    try :

        counterA = Counter(fake_title_train[each_element])
        counterB = Counter(title_train[each_element])
    except:
        print("Erroneous values. Discarded them")
        print(counterA)
        print(counterB)
    similarity_train.append(counter_cosine_similarity(counterA, counterB))
print("Similarity values for training calculated")

fake_title = []
title = []
label = []
for each in range(len(df_test)):
    fake_title.append(df_test.iloc[each][3])
    title.append(df_test.iloc[each][4])
print("Test Lists created")

similarity_val = []
for each_element in range(len(fake_title)):
    counterA = Counter(fake_title[each_element])
    counterB = Counter(title[each_element])
    similarity_val.append(counter_cosine_similarity(counterA, counterB))
print("Similarity values for Testing calculated")


x_train, y_train, x_test = similarity_train, label_train, similarity_val


x_train = np.array(x_train).reshape((-1, 1))
y_train = np.array(y_train).reshape(-1,1)

x_test = np.array(x_test).reshape((-1, 1))

neigh = KNeighborsClassifier(n_neighbors=505,weights ='uniform', p=2, metric ='euclidean')
neigh.fit(x_train, y_train)
y_pred = neigh.predict(x_test)

y_pred = y_pred.ravel()

pred = []
for each in range(len(y_pred)):
    if y_pred[each] == 0:
        pred.append('unrelated')
    elif y_pred[each] == 1:
        pred.append('agreed')
    else:
        pred.append('disagreed')

col  = pd.Series(df_test["id"], copy=True)
final=[]
for i in range(len(col)):
    final.append([col[i],pred[i]])
df_final=pd.DataFrame(final, columns=['id','label'])
df_final.to_csv('submission.csv', sep = ',',index = False)

print("Program terminated in  ", time.process_time() - start)