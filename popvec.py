__author__ = 'erwin'

import data_importer
import preparation
import logging
from gensim.models import word2vec

test, train, unlabeled_train = data_importer.test_train_unlabeled()

sentences = []
print 'parsing sentences from training set'
counter = 0
for review in train['review']:
    counter += 1
    print 'review %d from %d' % (counter, len(train['review']))
    sentences += preparation.review_to_sentences(review)

counter = 0
for review in unlabeled_train['review']:
    counter += 1
    print 'review %d from %d' % (counter, len(unlabeled_train['review']))
    sentences += preparation.review_to_sentences(review)

print len(sentences)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(messages)s',\
                    level=logging.INFO)

num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

print "Training model..."

model = word2vec.Word2Vec(sentences, workers=num_workers, \
                          size=num_features, min_count=min_word_count, \
                          window = context, sample = downsampling)

model.init_sims(replace=True)
model_name ='300features_40minwords_10context'
model.save(model_name)