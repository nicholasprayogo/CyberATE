
import logging
import codecs
import random
import copy
import nltk
from collections import defaultdict
from gensim import corpora, models, similarities
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import imblearn
from imblearn.under_sampling import RandomUnderSampler
import pickle
import re
import sys
import os
import time
from time import gmtime, strftime, localtime
import Preprocess
import Utility
from pymongo import MongoClient
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from tqdm import tqdm
from sklearn.utils import resample
import pandas as pd

from params import (
    local_vector_size,
    global_vector_size,
    local_global_vector_size,
    local_dim,
    global_dim
)

from helpers import (
    corpus2wordVSM,
    terms2list,
    get_glove_vector,
    word2localGlobalEmbeddings,
    term2localVector,
    term_list_2_ARFF,
    subtract_training,
    filter_terms_from_tool,
    word_list_2_dataset,
    get_array_with_class,
    create_dataset_separated,
    train_and_test,
    evaluate_tool,
    train_multichannel,
    generate_global_bert_embeddings_dict,
    create_multichannel_data
)



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

client = MongoClient('localhost', 27017)
db = client['cyber_ate']
collection = db['malware_results_new_multichannel']

correct_term_file = os.path.abspath("data/terms/cyber_CorrectTerms.txt")
non_term_file = os.path.abspath("data/terms/cyber_nonTerms.txt")

glove_dict_file_name="global_embedding_dicts/glove_global_dict.pkl"
glove_location = "global_embedding_dicts/glove.6B.50d.txt"

local_vec_generator = "w2v"
global_vec_generator = "glove"

if global_vec_generator=="glove":
    local_vector_size = 50

all_data_file = "data/malware_sentences.txt"
dataset_file = "data/annotated_words.txt"
bert_global_dict_file_name = "global_embedding_dicts/bert_global_dict.pkl"
txt_file_name = "data/malware_sentences.txt"
w2v_file = rf"w2v_models/w2v_{local_vector_size}_model_cyber_uncased"
local_bert_model_location = "./malware_bert_model"

global_vec_files_dict = {
    "glove": glove_dict_file_name,
    "bert": bert_global_dict_file_name
}

local_vec_files_dict = {
    "w2v": w2v_file,
    "bert": local_bert_model_location
}

correct_term_list = terms2list(correct_term_file)
single_word_term_list = [item for item in correct_term_list
                         if ((len(item.split())==1) and
                             ("-" not in item) and ("|" not in item))]

non_term_list = terms2list(non_term_file)
single_word_non_term_list = [item for item in non_term_list
                             if ((len(item.split())==1) and ("-" not in item) and ("|" not in item))]

if not os.path.exists(global_vec_files_dict[global_vec_generator]):

    corpus=all_data_file
    preprocessor = Preprocess.Preprocess()
    corpus_text = preprocessor.read_lower(corpus)
    text_tokens=preprocessor.word_tokenize(corpus_text)
    vocab = list(set(text_tokens))
    print("Original Vocab Size is:",len(vocab))
    print("Creating a frequency distribution")
    freq_dist=nltk.FreqDist(text_tokens)
    print("Done")
    filter=4
    print("Frequency filter is:",filter," Now filtering the vocabulary")
    vocab=[x for x in freq_dist.keys() if freq_dist[x]>filter]
    print("Done")
    print("The filtered vocabulary size is {}".format(len(vocab)))

    if global_vec_generator == "glove":
        global_dict = preprocessor.vocab_to_dict_for_large_vocab(vocab, glove_location,400000,True)
        global_dict_file_name = glove_dict_file_name

    elif global_vec_generator == "bert":
        global_dict = generate_global_bert_embeddings_dict(vocab)
        global_dict_file_name = bert_global_dict_file_name

    print("Created global vec dictionary for the vocabulary on {} the size of the dictionary is {}".format(strftime("%a, %d %b %Y %H:%M:%S", localtime()),len(bert_global_dict.keys())))
    preprocessor.pickle_file(global_dict, global_dict_file_name)

if global_vec_generator == "glove":
    global_dict = pickle.load( open( glove_dict_file_name, "rb" ) )
elif global_vec_generator == "bert":
    global_dict = pickle.load( open( bert_global_dict_file_name, "rb" ) )


if build_local_vectors:
    if local_vec_generator =="w2v":
        corpus2wordVSM(all_data_file, w2v_file, txt_file_name,
                       local_vector_size, 5, 5, 4)

    elif local_vec_generator == "bert":
        pass #TODO train model here instead of externally
        # generate_local_bert_embeddings()

if local_vec_generator =="w2v":
    local_model = models.Word2Vec.load(w2v_file) #CBOW
    local_tokenizer=None
elif local_vec_generator =="bert":
    local_model = DistilBertModel(DistilBertConfig()).from_pretrained(local_bert_model_location)
    local_tokenizer = DistilBertTokenizer.from_pretrained(local_bert_model_location)

util=Utility.Utility()
preprocessor = Preprocess.Preprocess()

data_dir = os.path.abspath(f"data/multichannel_{global_vec_generator}_global_{local_vec_generator}_local/")

if global_vec_generator =="glove":
    global_dim = 50
else:
    global_dim = 768

if not os.path.exists(data_dir):

    with open(dataset_file, "r") as f:
        vocab = [i.rstrip('\n') for i in f.readlines()]

    datapoints = []
    data_classes = []

    datapoints_local = []
    datapoints_global = []
    # print(len(vocab))
    vocab_copy=copy.deepcopy(vocab)

    for word in tqdm(vocab_copy):

       global_vec, local_vec = util.new_word2localGlobalEmbeddings_models_separated(word, global_vec_generator, global_dict, global_dim, local_vec_generator, local_model, global_dim, local_tokenizer=local_tokenizer)
       datapoints_global.append(np.array(global_vec, np.float32))
       datapoints_local.append(np.array(local_vec , np.float32))

       if word in correct_term_list:
           label = 1
       else:
           label = 0

       data_classes.append(label)

    datapoints_local, datapoints_global, data_classes = create_multichannel_data(datapoints_local, datapoints_global, data_classes)

    data_set_local = [datapoints_local, data_classes]
    data_set_global = [datapoints_global, data_classes]
    data_set_local_reshaped = [datapoints_local, data_classes.reshape(len(data_classes),)]
    data_set_global_reshaped = [datapoints_global, data_classes.reshape(len(data_classes),)]

    os.mkdir(data_dir)

    preprocessor.pickle_file(data_set_local, os.path.join(data_dir, "the_dataset_local.pkl"))
    preprocessor.pickle_file(data_set_local_reshaped, os.path.join(data_dir, "the_dataset_local_reshaped.pkl"))
    preprocessor.pickle_file(data_set_global, os.path.join(data_dir, "the_dataset_global.pkl"))
    preprocessor.pickle_file(data_set_global_reshaped, os.path.join(data_dir, "the_dataset_global_reshaped.pkl"))


embedding_name = "bert"

n_epochs = 50
batch_size = 64
hidden_dim = 16

train_multichannel(local_vec_generator, global_vec_generator, collection, data_dir, n_epochs, batch_size, hidden_dim)
