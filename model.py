import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import tensorflow as tf
import prepo as prepo
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/masterdata.csv")
data['tokens'] = data.apply(
    lambda data: nltk.word_tokenize(data['content']), axis=1)

# Split data Test
data_train, data_test = train_test_split(data, test_size=0.2, shuffle=True)
# Training
all_training_words = [word for tokens in data_train["tokens"]
                      for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" %
      (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))
print("Split Data Train finish..............")

# Split Data Test
all_test_words = [word for tokens in data_test["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data_test["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
print("%s words total, with a vocabulary size of %s" %
      (len(all_test_words), len(TEST_VOCAB)))
print("Max sentence length is %s" % max(test_sentence_lengths))
print("Split Data Test finish..............")

data_train = data_train.to_csv('data_train.csv')
