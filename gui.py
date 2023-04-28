# from crypt import methods
import csv
import pandas as pd
from distutils import text_file
from json import load
from shutil import register_unpack_format
import pickle
from tabnanny import verbose
from typing import final
import nltk
from distutils.log import debug
from operator import sub
import numpy as np
import prepo_lib as prepo
from flask import Flask, request, jsonify, render_template
from flask_paginate import Pagination, get_page_parameter
from keras_preprocessing.sequence import pad_sequences
from keras.models import model_from_json, load_model
nltk.download('punkt')

app = Flask(__name__, static_url_path='/static')

tokenizer = pickle.load(open('data/model/Saved_Tokenize.pickle', 'rb'))

# Skipgram
json_file = open("data/model/saved_model_sentimen_sg.json", 'r')
model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("data/model/saved_model_sentimen_sg.h5")
# model = load_model("data/model_sg5.h5")  # notenk

# CBOW
json_file_c = open("data/model/saved_model_sentimen_cbow.json", 'r')
model_json_c = json_file_c.read()
model_c = model_from_json(model_json_c)
model_c.load_weights("data/model/saved_model_sentimen_cbow.h5")
# model_c = load_model("data/model_cbow5.h5")

class_category = ['POSITIF', 'NETRAL', 'NEGATIF']

MAX_SEQUENCE_LENGTH = 100


@ app.route('/')
def home():
    return render_template("index.html", debug=True)


@ app.route('/index.html')
def dash():
    return render_template("index.html", debug=True)


@ app.route('/datalatih.html')
def datalatih():
    page = request.args.get(get_page_parameter(), type=int, default=1)
    with open('data/data_train_label.csv', encoding='unicode_escape') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        first_line = True
        dataset = []
        for row in data:
            if not first_line:
                dataset.append({
                    "content": row[1],
                    "sentimen": row[6],
                    "label": row[2]
                })
            else:
                first_line = False

    pagination = Pagination(
        page=page, total=len(dataset), record_name='datasets')
    per_page = pagination.per_page
    if page == 1 or not page:
        dataset_paginatiom = dataset[:per_page]
    else:
        dataset_paginatiom = dataset[(page*per_page)-1: ((page+1)*per_page)-1]
    return render_template("datalatih.html", len=len(dataset), dataset=dataset_paginatiom, pagination=pagination, debug=True)


@ app.route('/dataklasifikasi.html', methods=["POST", "GET"])
def dataklasifikasi():
    page = request.args.get(get_page_parameter(), type=int, default=1)
    with open('data/hasil_prediksi.csv', encoding='unicode_escape') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        first_line = True
        dataset = []
        for row in data:
            if not first_line:
                dataset.append({
                    "content": row[1],
                    "label": row[2],
                    "predict": row[3]
                })
            else:
                first_line = False

    pagination = Pagination(
        page=page, total=len(dataset), record_name='datasets')
    per_page = pagination.per_page
    if page == 1 or not page:
        dataset_paginatiom = dataset[:per_page]
    else:
        dataset_paginatiom = dataset[(page*per_page)-1: ((page+1)*per_page)-1]

    return render_template("dataklasifikasi.html", len=len(dataset), dataset=dataset_paginatiom, pagination=pagination, debug=True)


@ app.route('/data2021.html', methods=["POST", "GET"])
def data2021():
    page = request.args.get(get_page_parameter(), type=int, default=1)
    with open('data/hasil_prediksi2021.csv', encoding='unicode_escape') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        first_line = True
        dataset = []
        for row in data:
            if not first_line:
                dataset.append({
                    "content": row[1],
                    "label": row[2],
                    "predict": row[3]
                })
            else:
                first_line = False

    pagination = Pagination(
        page=page, total=len(dataset), record_name='datasets')
    per_page = pagination.per_page
    if page == 1 or not page:
        dataset_paginatiom = dataset[:per_page]
    else:
        dataset_paginatiom = dataset[(page*per_page)-1: ((page+1)*per_page)-1]

    return render_template("data2021.html", len=len(dataset), dataset=dataset_paginatiom, pagination=pagination, debug=True)


@ app.route('/datacrawler.html', methods=["POST", "GET"])
def datacrawler():
    page = request.args.get(get_page_parameter(), type=int, default=1)
    with open('data/data13k.csv', encoding='unicode_escape') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        first_line = True
        dataset = []
        for row in data:
            if not first_line:
                dataset.append({
                    "content": row[0]
                })
            else:
                first_line = False

    pagination = Pagination(
        page=page, total=len(dataset), record_name='datasets')
    per_page = pagination.per_page
    if page == 1 or not page:
        dataset_paginatiom = dataset[:per_page]
    else:
        dataset_paginatiom = dataset[(page*per_page)-1: ((page+1)*per_page)-1]
    return render_template("datacrawler.html", len=len(dataset), dataset=dataset_paginatiom, pagination=pagination, debug=True)


@ app.route('/grafikuji.html')
def grafikuji():
    return render_template("grafikuji.html", debug=True)


@ app.route('/sentimen.html')
def ceksentimen():
    # dataset = []
    return render_template("sentimen.html",  debug=True)


@ app.route('/hasilsentimen.html', methods=["GET"])
def hasilsentimen():
    subject = request.args.get("sub")
    subject = [subject]

    # # casefolding
    test_casefolding = []
    for i in range(len(subject)):
        if subject[i].islower() == True:
            test_casefolding = subject[i]
        else:
            test_casefolding = prepo.case_folding(subject[i])

    casefolding = test_casefolding

    # Cleaning Data
    removenum = prepo.cleaning(casefolding)
    # Slangword
    slangword_ = prepo.slangword(removenum)
    # # Stemming
    stemming_ = prepo.stemming(slangword_)
    # Negation Handling
    if 'tidak' in stemming_:
        negation_ = prepo.ganti_negasi(stemming_)
    else:
        negation_ = stemming_
    # # Stopword
    remove_stop_words = prepo.removeStopword(negation_)
    remove_stop_words1 = prepo.removeStopword(stemming_)
    # Tokenize
    hasil_token = prepo.tokenize(remove_stop_words)
    hasil_token1 = prepo.tokenize(remove_stop_words1)
    text = str(hasil_token)
    text1 = str(hasil_token1)
    # Text Final
    final = prepo.text_final(hasil_token)
    final1 = prepo.text_final(hasil_token1)
    text_final_ = str(final)
    text_final_1 = str(final1)

    # CNN + Negation
    sequences = tokenizer.texts_to_sequences(
        [text])  # untuk menentukan urutan kata
    print('sequences :', sequences)
    test_cnn_data = pad_sequences(
        sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x_test = test_cnn_data
    print('x_test :', x_test)
    predictions = model.predict(x_test)
    prob_sg = predictions
    class_sg = class_category[predictions.argmax()]

    # CNN + Non Negation
    sequences1 = tokenizer.texts_to_sequences([text1])
    print('sequences :', sequences1)
    test_cnn_data1 = pad_sequences(
        sequences1, maxlen=MAX_SEQUENCE_LENGTH)
    x_test1 = test_cnn_data1
    print('x_test :', x_test1)
    predictions1 = model.predict(x_test1)
    prob_sg_non = predictions1
    class_sg_non = class_category[predictions1.argmax()]

    return render_template("hasilsentimen.html",
                           subject=subject,
                           casefolding=casefolding,
                           removenum=removenum,
                           slangword_=slangword_,
                           negation_=negation_,
                           hasil_token=hasil_token,
                           remove_stop_words=remove_stop_words,
                           stemming_=stemming_,
                           text_final_=text_final_,
                           text_final_1=text_final_1,
                           prob_sg=prob_sg,
                           class_sg=class_sg,
                           prob_sg_non=prob_sg_non,
                           class_sg_non=class_sg_non,
                           debug=True)


@app.route('/sentimen_non.html', methods=["GET"])
def sentimen_non():
    return render_template("sentimen_non.html", debug=True)


@app.route('/hasilsentimen_non.html', methods=["POST", "GET"])
def hasilsentimen_non():
    subject = request.args.get("sub")
    subject = [subject]

    # # casefolding
    test_casefolding = []
    for i in range(len(subject)):
        if subject[i].islower() == True:
            test_casefolding = subject[i]
        else:
            test_casefolding = prepo.case_folding(subject[i])

    casefolding = test_casefolding

    # Cleaning Data
    removenum = prepo.cleaning(casefolding)
    # Slangword
    slangword_ = prepo.slangword(removenum)
    # # Stemming
    stemming_ = prepo.stemming(slangword_)
    # Negation Handling
    if 'tidak' in stemming_:
        negation_ = prepo.ganti_negasi(stemming_)
    else:
        negation_ = stemming_
    # # Stopword
    remove_stop_words = prepo.removeStopword(negation_)
    remove_stop_words1 = prepo.removeStopword(stemming_)
    # Tokenize
    hasil_token = prepo.tokenize(remove_stop_words)
    hasil_token1 = prepo.tokenize(remove_stop_words1)
    text = str(hasil_token)
    text1 = str(hasil_token1)
    # Text Final
    final = prepo.text_final(hasil_token)
    final1 = prepo.text_final(hasil_token1)
    text_final_ = str(final)
    text_final_1 = str(final1)

    # CNN + Negation
    sequences = tokenizer.texts_to_sequences(
        [text])  # untuk menentukan urutan kata
    print('sequences :', sequences)
    test_cnn_data = pad_sequences(
        sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x_test = test_cnn_data
    print('x_test :', x_test)
    predictions = model_c.predict(x_test)
    prob_cbow = predictions
    class_cbow = class_category[predictions.argmax()]

    # CNN + Non Negation
    sequences1 = tokenizer.texts_to_sequences([text1])
    print('sequences :', sequences1)
    test_cnn_data1 = pad_sequences(
        sequences1, maxlen=MAX_SEQUENCE_LENGTH)
    x_test1 = test_cnn_data1
    print('x_test :', x_test1)
    predictions1 = model_c.predict(x_test1)
    prob_cbow_non = predictions1
    class_cbow_non = class_category[predictions1.argmax()]

    return render_template("hasilsentimen_non.html",
                           subject=subject,
                           casefolding=casefolding,
                           removenum=removenum,
                           slangword_=slangword_,
                           hasil_token=hasil_token,
                           remove_stop_words=remove_stop_words,
                           negation_=negation_,
                           stemming_=stemming_,
                           text_final_=text_final_,
                           text_final_1=text_final_1,
                           prob_cbow=prob_cbow,
                           class_cbow=class_cbow,
                           prob_cbow_non=prob_cbow_non,
                           class_cbow_non=class_cbow_non,
                           debug=True)


if __name__ == "__main__":
    app.run(debug=True)
