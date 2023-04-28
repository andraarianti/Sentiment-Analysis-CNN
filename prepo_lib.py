from inspect import ClassFoundException
import os
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import tensorflow as tf
import re
import numpy as np
import pandas as pd
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from keras.preprocessing.text import Tokenizer


def case_folding(tokens):
    return tokens.lower()


def cleaning(content):
    # replace non ASCII char
    content = re.sub(r'[^\x00-\x7f]', r'', content)
    content = re.sub(r'(\\u[0-912345678A-Fa-f]+)', r'', content)
    content = re.sub(r"[^A-Za-z0123456789^,!.\/'+-=]", " ",  content)
    content = re.sub(r'\\u\w\w\w\w', '', content)
    # Remove simbol, angka dan karakter aneh
    content = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", content)
    return content


def open_kamus_prepro(x):
    kamus = {}
    with open(x, 'r') as file:
        for line in file:
            text = line.replace("'", "").split(':')
            kamus[text[0].strip()] = text[1].rstrip('\n').lstrip()
    return kamus


def slangword(content):
    df_slang = pd.read_csv('data/slangword.csv', on_bad_lines='skip')
    res = ''
    # += operator lets you add two values together and assign the resultant value to a variable.
    for item in content.split():
        if item in df_slang.slang.values:
            res += df_slang[df_slang['slang'] == item]['formal'].iloc[0]
        else:
            res += item
        res += ' '
    return res


# stopword
stopwords = StopWordRemoverFactory().get_stop_words()


def removeStopword(content):
    # stopwords removing
    word_lst = word_tokenize(content)
    word_lst = [word for word in word_lst if word not in stopwords]

    content = ' '.join(word_lst)
    return content


# negasi
kamus_negasi = open_kamus_prepro('data/negasi.txt')


def ganti_negasi(w):
    w_splited = w.split(' ')
    if 'tidak' in w_splited:
        # enumerate : method adds a counter to an iterable and returns it in a form of enumerating object
        for i, k in enumerate(w_splited):
            if k in kamus_negasi and w_splited[i-1] == 'tidak':
                w_splited[i] = kamus_negasi[k]

    return ' '.join(w_splited)


# Stem
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def stemming(data):
    data = data.lower()
    data = stemmer.stem(data)
    return data


def tokenize(data):
    data = nltk.tokenize.word_tokenize(str(data))
    return data


def text_final(data):
    data = TreebankWordDetokenizer().detokenize(data)
    return data


def preprocessing_text(content):
    content = case_folding(content)
    content = cleaning(content)
    content = slangword(content)
    content = stemming(content)
    content = removeStopword(content)
    content = ganti_negasi(content)
    content = tokenize(content)
    content = text_final(content)
    return content


text = "kartu tidak bisa dibaca bagaimana nihh"
token = tokenize(text)
