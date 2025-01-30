import numpy as np
import re

stopwords = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"}

def tokenize(sentence):

    return [word for word in re.findall(r'\b\w+\b', sentence.lower()) if word not in stopwords]

def stem(word):
    suffixes = ["ing", "ed", "es", "s", "ment"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix):
            return word[:-len(suffix)]
    return word

def bag_of_words(tokenized_sentence, words):

    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
