
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
from transformers import BertTokenizer, BertConfig, BertModel
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
    generate_global_bert_embeddings_dict
)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

client = MongoClient('localhost', 27017)
db = client['cyber_ate']
collection = db['cyber_final_new']

correct_term_file = os.path.abspath("data/terms/common_CorrectTerms.txt")
non_term_file = os.path.abspath("data/terms/common_nonTerms.txt")

glove_dict_file_name="global_embedding_dicts/glove_global_dict.pkl"

glove_location = "global_embedding_dicts/glove.6B.50d.txt"

implementation = "bert_global_w2v_local"

# TODO pass config as params
local_vec_generators = ["bert_raw", "w2v", "bert_finetune"]
global_vec_generators = ["glove", "bert"]

all_data_file = "data/malware_sentences_raw.txt"
dataset_file = "data/annotated_words_new.txt"
bert_global_dict_file_name = "global_embedding_dicts/bert_global_dict.pkl"
txt_file_name = "data/malware_sentences_raw.txt"


for local_vec_generator in local_vec_generators:
    for global_vec_generator in global_vec_generators:
        if local_vec_generator == "bert_raw":
            local_bert_model_location = "./cyber_bert_model_raw"

        elif local_vec_generator == "bert_finetune":
            local_bert_model_location = "./cyber_bert_model_finetune_raw"

        elif local_vec_generator == "bert_refined":
            local_bert_model_location = "./cyber_bert_model_raw_new"

        else:
            local_bert_model_location = "./cyber_bert_model_raw"

        if local_vec_generator=="w2v":
            local_vector_size = 100
            local_dim = 100
        else:
            local_vector_size = 768
            local_dim = 768

        global_vec_files_dict = {
            "glove": glove_dict_file_name,
            "bert": bert_global_dict_file_name
        }

        w2v_file = rf"w2v_models/w2v_{local_vector_size}_model_cyber_uncased"

        local_vec_files_dict = {
            "w2v": w2v_file,
            "bert": local_bert_model_location,
            "bert_raw": local_bert_model_location,
            "bert_finetune":local_bert_model_location,
            "bert_refined": local_bert_model_location
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

            print("Created global vec dictionary for the vocabulary on {} the size of the dictionary is {}".format(strftime("%a, %d %b %Y %H:%M:%S", localtime()),len(global_dict.keys())))
            preprocessor.pickle_file(global_dict, global_dict_file_name)

        global_dict = pickle.load( open( global_vec_files_dict[global_vec_generator], "rb" ) )

        if not os.path.exists(local_vec_files_dict[local_vec_generator]):
            if local_vec_generator =="w2v":
                corpus2wordVSM(all_data_file, w2v_file, txt_file_name,
                               local_vector_size, 5, 5, 4)

            elif local_vec_generator == "bert" or local_vec_generator == "bert_raw":
                pass #TODO train model here instead of externally
                # generate_local_bert_embeddings()

        if local_vec_generator =="w2v":
            local_model = models.Word2Vec.load(w2v_file) #CBOW
            local_tokenizer = None
        elif local_vec_generator in ["bert", "bert_raw", "bert_finetune", "bert_refined"]:
            local_model = BertModel(BertConfig()).from_pretrained(local_bert_model_location)
            local_tokenizer = BertTokenizer.from_pretrained(local_bert_model_location)

        util=Utility.Utility()
        preprocessor = Preprocess.Preprocess()

        data_dir = os.path.abspath(f"final_data_new/new_{global_vec_generator}_global_{local_vec_generator}_local/")

        if global_vec_generator =="glove":
            global_dim = 50
        else:
            global_dim = 768

        if not os.path.exists(data_dir):

            with open(dataset_file, "r") as f:
                vocab = [i.rstrip('\n') for i in f.readlines()]

            datapoints = []
            data_classes = []

            vocab_copy=copy.deepcopy(vocab)

            for word in tqdm(vocab_copy):
                if lcoal_only:
                    datapoint = util.new_word2localGlobalEmbeddings_models_localonly(word, global_vec_generator, global_dict, global_dim,
                                                   local_vec_generator, local_model, local_dim, local_tokenizer=local_tokenizer)
                else:
                   datapoint = util.new_word2localGlobalEmbeddings_models(word, global_vec_generator, global_dict, global_dim,
                                                  local_vec_generator, local_model, local_dim, local_tokenizer=local_tokenizer)

                datapoints.append(np.array(datapoint, np.float32))

                if word in correct_term_list:
                   label = 1
                else:
                   label = 0

                data_classes.append(label)

            X = np.array(datapoints)
            Y = np.array(data_classes)

            df = pd.DataFrame({"X":datapoints,"Y":list(data_classes)})

            df_minority = df[df["Y"]==1]
            df_majority = df[df["Y"]==0]

            df_minority_upsampled = resample(df_minority,
                                         replace=True,
                                         n_samples=int(df_majority.size/2),
                                         random_state=2)

            df_upsampled = pd.concat([df_minority_upsampled,df_majority]).reset_index()

            datapoints = df_upsampled["X"]
            data_classes = df_upsampled["Y"]

            new_datapoints =[]
            new_data_classes = []

            # convert back to list cus pandas series lose dimensionality

            for i in datapoints:
                new_datapoints.append(list(i))

            for i in data_classes:
                new_data_classes.append(int(i))

            datapoints = np.array(new_datapoints)
            data_classes = np.array(data_classes)

            data_set=[datapoints, data_classes]

            data_set_reshaped = [datapoints, data_classes.reshape(len(data_classes),)]

            os.mkdir(data_dir)
            preprocessor.pickle_file(data_set, os.path.join(data_dir, "the_dataset.pkl"))
            preprocessor.pickle_file(data_set_reshaped, os.path.join(data_dir, "the_dataset_reshaped.pkl"))

        embedding_name = "bert"

        train_and_test("mlp", 1, local_vec_generator, global_vec_generator, collection, data_dir)
        train_and_test("svm", 1, local_vec_generator, global_vec_generator, collection, data_dir)
        train_and_test("logistic",1, local_vec_generator, global_vec_generator, collection, data_dir)
