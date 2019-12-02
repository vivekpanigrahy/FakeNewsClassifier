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
from sklearn.metrics import precision_recall_fscore_support as score

start = time.process_time()

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)


df_train = pd.read_csv("./data/train_mod.csv")
df_val = pd.read_csv("./data/validation_mod.csv")

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
        print(counterA)
        print(counterB)
    similarity_train.append(counter_cosine_similarity(counterA, counterB))
print("Similarity values for training calculated")

fake_title = []
title = []
label = []
for each in range(len(df_val)):
    fake_title.append(df_val.iloc[each][3])
    title.append(df_val.iloc[each][4])
    if df_val.iloc[each][5] == 'unrelated':  
        label.append(0)
        continue
    elif df_val.iloc[each][5] == 'agreed':
        label.append(1)
        continue
    else:
        label.append(2)
print("Test Lists created")

similarity_val = []
for each_element in range(len(fake_title)):
    counterA = Counter(fake_title[each_element])
    counterB = Counter(title[each_element])
    similarity_val.append(counter_cosine_similarity(counterA, counterB))
print("Similarity values for Testing calculated")

x_train, y_train, x_test, y_test = similarity_train, label_train, similarity_val, label  # Uncomment this part when you run the validation set

x_train = np.array(x_train).reshape((-1, 1))
y_train = np.array(y_train).reshape(-1,1)

x_test = np.array(x_test).reshape((-1, 1))
y_test = np.array(y_test).reshape(-1,1)

neigh = KNeighborsClassifier(n_neighbors=40,weights ='uniform', p=2, metric ='euclidean')
neigh.fit(x_train, y_train)
y_pred = neigh.predict(x_test)

precision, recall, fscore, support = score(y_test, y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

y_pred = y_pred.ravel()

pred = []
for each in range(len(y_pred)):
    if y_pred[each] == 0:
        pred.append('unrelated')
    elif y_pred[each] == 1:
        pred.append('agreed')
    else:
        pred.append('disagreed')
# df_val['label'] = pred
print(df_val.head())

print(y_test)
print(y_pred[0:10])

acc = accuracy_score(y_test, y_pred)
print("Accuracy = ", acc)

print("Program terminated in  ", time.process_time() - start)