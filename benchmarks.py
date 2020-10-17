
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
    evaluate_tool_new
)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

client = MongoClient('localhost', 27017)
db = client['cyber_ate']
collection = db['cyber_final_new']

create_dataset = True
create_global_vectors = False
split_stratified = False
build_local_vectors = False

correct_term_file = os.path.abspath("data/terms/common_CorrectTerms.txt")
non_term_file = os.path.abspath("data/terms/common_nonTerms.txt")

all_data_file = "data/malware_sentences.txt"
dataset_file = "data/annotated_words_new.txt"
txt_file_name = "data/malware_sentences.txt"

correct_term_list = terms2list(correct_term_file)
single_word_term_list = [item for item in correct_term_list
                         if ((len(item.split())==1) and
                             ("-" not in item) and ("|" not in item))]

non_term_list = terms2list(non_term_file)
single_word_non_term_list = [item for item in non_term_list
                             if ((len(item.split())==1) and ("-" not in item) and ("|" not in item))]


termoStat_extractions = terms2list(os.path.abspath("data/benchmarks/TermoStat.txt"))
termoStat_extractions = [re.split(r"\s+",item)[0] for item in termoStat_extractions if len(re.split(r"\s+",item))>=2]

termoStat_single_word_list = [item for item in termoStat_extractions
                                         if ((len(item.split())==1) and
                                             ("-" not in item)and ("|" not in item))]

new_result = evaluate_tool_new(termoStat_extractions, correct_term_list, non_term_list)
collection.insert_one(new_result)
