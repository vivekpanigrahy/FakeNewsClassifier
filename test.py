import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
import csv
from collections import Counter
import math
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)


df_train = pd.read_csv("./data/train_new.csv")
df_val = pd.read_csv("./data/val_new.csv")

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
    
    # label_train.append(df_train.iloc[each_row][5])
print("Training Lists created")
print(label_train)

similarity_train = []

for each_element in range(len(fake_title_train)):
    # print(fake_title_train[each_element])
    # print(title_train[each_element])
    try :

        counterA = Counter(fake_title_train[each_element])
        counterB = Counter(title_train[each_element])
    except:
        print(counterA)
        print(counterB)
    similarity_train.append(counter_cosine_similarity(counterA, counterB))
print("Similarity values for training calculated")
print(similarity_train[:10])

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
    # label.append(df_val.iloc[each][5])
print("Validation Lists created")
print(label)

similarity_val = []
for each_element in range(len(fake_title)):
    counterA = Counter(fake_title[each_element])
    counterB = Counter(title[each_element])
    similarity_val.append(counter_cosine_similarity(counterA, counterB))
print("Similarity values for validation calculated")
print(similarity_val[:10])

x_train, y_train, x_test, y_test = similarity_train, label_train, similarity_val, label

regressor = LinearRegression()
x_train = np.array(x_train).reshape((-1, 1))
y_train = np.array(y_train).reshape(-1,1)

x_test = np.array(x_test).reshape((-1, 1))
y_test = np.array(y_test).reshape(-1,1)

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
df = pd.DataFrame([{'Actual': y_test, 'Predicted': y_pred}])
print(df)

# print("Training set :")
# print(df_train.head(10))