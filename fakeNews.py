from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
import numpy as np 
import csv
import sys
import pickle

csv.field_size_limit(sys.maxsize)
csv_path = "data/news.csv"
with open(csv_path, 'r') as csv_file: 
    database = list(csv.reader(csv_file))
vectorizer = CountVectorizer()
text = database[1:4001]
text = map(lambda x:x[2], text)
text_vectors = vectorizer.fit_transform(text)
test = database[4001:]
test = map(lambda x: x[2], test)
test_vectors = vectorizer.transform(test)
y_targets = map(lambda x: True if x[3] == 'FAKE' else False, database)
test_vectors =  vectorizer.transform(test)
clf = MultinomialNB()
clf.fit(text_vectors, y_targets[1:4001])
y_predict = clf.predict(test_vectors)
count = 0
count_true = 0
count_false = 0
y_target = y_targets[4001:] 
for x in range(len(y_predict)):
    if y_predict[x] == y_target[x]:
        count += 1
print count*100/float(len(y_predict))
#print(database.shape)
with open('model/fake_vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('model/fake_news.pickle', 'wb') as f:
    pickle.dump(clf, f)
