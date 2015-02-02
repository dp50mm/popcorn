__author__ = 'erwin'
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import math

def review_to_words(raw_review, remove_stopwords=False):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]"," ",review_text)
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (" ".join(words))

def review_to_wordlist(raw_review, remove_stopwords=False):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]"," ",review_text)
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


def root_features(feature_array):
    output_array = []
    for feature_vector in feature_array:
        output_array.append([math.sqrt(x) for x in feature_vector])
    return output_array