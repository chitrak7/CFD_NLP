import numpy as np 
import csv
import sys
import pickle
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

csv.field_size_limit(sys.maxsize)

vectorizer = CountVectorizer()
temp = []


csv_path = "data/news3.csv"
with open(csv_path, 'r') as csv_file: 
    database = list(csv.reader(csv_file))

for i in range(10):
    print(database[i])
for rows in database:
    if (rows[4]=='1'):
        temp.append([rows[1], -1])
    elif (rows[4]=='2'):
        temp.append([rows[1], 1])
    else:
        temp.append([rows[1], 0])
database = temp
database1 = map(lambda x: x[0], database)
vectorizer.fit(database1)
train = database[0:4000]
test = database[-2000:]

trainX =  map(lambda x: x[0], train)
trainY =  map(lambda x: x[1], train)

testX =  map(lambda x: x[0], test)
testY =  map(lambda x: x[1], test)

testX = vectorizer.transform(testX)
trainX = vectorizer.transform(trainX)

clf = MultinomialNB()
clf.fit(trainX, trainY)
pred = clf.predict(testX)
count = 0
for x in range(len(pred)):
    if pred[x] == testY[x]:
        count += 1

print count*100/float(len(pred))

with open('model/bias_news.pickle', 'wb') as f:
    pickle.dump(clf, f)

with open('model/bias_vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)
