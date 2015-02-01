__author__ = 'erwin'
import pandas as pd

def test_train_unlabeled():
    train = pd.read_csv('data/labeledTrainData.tsv',header = 0, \
                    delimiter = '\t', quoting = 3)
    test = pd.read_csv('data/testData.tsv', header = 0, \
                       delimiter="\t", quoting = 3)
    unlabeled_train = pd.read_csv("data/unlabeledTrainData.tsv",header=0, \
                                  delimiter ="\t", quoting = 3)
    print "Read %d labeled train reviews, %d labeled test reviews, "\
        "and %d unabeled reviews \n" % (train['review'].size, test['review'].size, unlabeled_train['review'].size)
    return test, train, unlabeled_train