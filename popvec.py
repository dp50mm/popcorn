__author__ = 'erwin'

import data_importer
import preparation


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

