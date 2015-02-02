__author__ = 'erwin'


import numpy as np
import pandas as pd
from preparation import review_to_words, root_features
from sklearn.feature_extraction.text import CountVectorizer
import math

from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t",quoting=3)

num_reviews = train['review'].size

clean_train_reviews = []

print 'cleaning training set reviews'
for i in xrange(0, num_reviews):
    print 'review %d of %d\n' % (i+1, num_reviews)
    clean_train_reviews.append(review_to_words(train['review'][i],remove_stopwords=True))


vectorizer = CountVectorizer(analyzer= "word", \
                             tokenizer = None,  \
                             preprocessor = None, \
                             stop_words = None, \
                             max_features = 10000)

print 'fit transform'
train_data_features = vectorizer.fit_transform(clean_train_reviews)
print 'data to arrays'
train_data_features = train_data_features.toarray()

print 'vocabullary vector'
vocab = vectorizer.get_feature_names()

dist = np.sum(train_data_features, axis=0)

for tag, count in zip(vocab,dist):
    print count, tag

forest = RandomForestClassifier(n_estimators = 200)
print 'fiting classifier'
#forest = forest.fit(train_data_features, train['sentiment'])
forest_root = forest.fit(root_features(train_data_features), train['sentiment'])
print 'opening testdata'
test = pd.read_csv("data/testData.tsv",header=0,delimiter="\t", \
                   quoting=3)

print test.shape

num_test_reviews = len(test['review'])
clean_test_reviews = []

print 'cleaning test reviews'
for i in xrange(0,num_test_reviews):
    if((i+1) % 1000 == 0):
        print 'review %d of %d' %(i+1, num_reviews)
    clean_review = review_to_words(test['review'][i])
    clean_test_reviews.append(clean_review)

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
print 'predicting with forest'
result = forest_root.predict(root_features(test_data_features))

output = pd.DataFrame(data={'id':test['id'],'sentiment':result})

output.to_csv("data/Bag_of_Words_model_root.csv",index=False, quoting=3)



