import logging
import codecs
import random
import copy
import nltk
from collections import defaultdict

from torch.nn.modules.container import Sequential
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

import Preprocess
import Utility
from multichannel import (
    CustomDataset,
    Multichannel,
    NeuralNetwork
)
import pandas as pd 

from sklearn.utils import shuffle

# import tensorflow as tf
# from tensorflow.keras.layers import Dense 
# from tensorflow.keras import Sequential, Input 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.metrics import Precision, Recall
def generate_global_bert_embeddings_dict(vocab):
    embeddings_list = []
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for vocab_token in tqdm(vocab):
        input_ids = torch.tensor(tokenizer.encode(vocab_token)).unsqueeze(0)
        hidden_states = model(input_ids)[2]
        token_embeddings = torch.stack(hidden_states, dim=0) # add 1 more dimension
        token_embeddings = torch.squeeze(token_embeddings, dim=1) # remove "batch" dimension
        token_embeddings = token_embeddings.permute(1,0,2) # put tokens dimension in front
        bert_token = token_embeddings[1] # select the middle token for the target word
        sum_vec = torch.sum(bert_token[-4:], dim=0) # will have 768
        embedding_vector = [vocab_token, list(sum_vec)]
        embeddings_list.append(embedding_vector)

    vectors_dict = dict()
    for embedding in embeddings_list:
        key = embedding[0]
        value = [np.float32(x) for x in embedding[1:]]
        vectors_dict.update({key:value})

    return vectors_dict

def corpus2wordVSM(corpus_file_name, embeddings_file_name=r"embeddings.w2v", txt_file_name= "sent_line_math_corpus.txt",
                   feature_vec_size=100, window_size=5, minimum_count=5, num_of_virtual_cores=4, skipGram=0):
    math_corpus_file = codecs.open(corpus_file_name, "r", "utf-8")
    math_corpus_text = math_corpus_file.read().lower()
    math_corpus_file.close()
    math_corpus_sanitized_sentence_tokenized = nltk.tokenize.sent_tokenize(math_corpus_text)
    sent_line_math_corpus_file = open(txt_file_name, "a")
    for sentence in math_corpus_sanitized_sentence_tokenized:
        sent_line_math_corpus_file.write("\n")
        word_tokens = nltk.tokenize.word_tokenize(sentence)
        for word in word_tokens:
            # if isinstance(word, unicode):
            sent_line_math_corpus_file.write(word.lower()+" ")
    sent_line_math_corpus_file.close()
    sentences = models.word2vec.LineSentence(txt_file_name)
    model = models.Word2Vec(sentences, size=feature_vec_size, window=window_size, min_count=minimum_count, workers=num_of_virtual_cores, sg=skipGram)
    #if the parameter sg=0 (defult) is changed to sg=1,
    #the model will be skip-gram as opposed to CBOW


    model.save(embeddings_file_name)

def terms2list(terms_file_name):
    terms_file = codecs.open(terms_file_name, "r", "utf-8")
    term_list = terms_file.read().splitlines()
    terms_file.close()
    return term_list

def get_glove_vector(word, glove_file_name, size_of_not_found_vector=50):
    with codecs.open(glove_file_name, "r", "utf-8") as glove_file:
        vector = []
        returning_vector = []
        for line in glove_file:
            lineList = line.split()
            if lineList[0] == word:
                vector = lineList[1:]
        if len(vector)!=0:
            returning_vector = vector
        else: returning_vector = [0]*size_of_not_found_vector
        #code for random vector instead of zeros vector:
            #RandomVector = [random.randint(LowerBound,UpperBound) for i in range(VectorSize)]
    return returning_vector

def word2localGlobalEmbeddings(word, glove_file_name, glove_dimension,
                               local_VSM_dimension, local_VSM_model):
    global_vec = get_glove_vector(word,glove_file_name, glove_dimension)
    local_vec = term2localVector(word, local_VSM_dimension, local_VSM_model)
    #note: model has to be created by "corpus2wordVecCBOW" before the above line and
    #hence this function can work
    local_global_embeddings = local_vec+global_vec
    return local_global_embeddings

def term2localVector(term, size_of_not_found_vector, local_vsm_model):
    vector = []
    if local_vsm_model.vocab.has_key(term):
        vector = local_vsm_model[term].tolist() #since it's a numpy vector and needs to be
        #turned into a simple list to work for my algorithm
        returning_vector = vector
    else: vector = [0]*size_of_not_found_vector
    #code for random vector instead of zeros vector:
            #RandomVector = [random.randint(LowerBound,UpperBound) for i in range(VectorSize)]
    return vector

def term_list_2_ARFF(local_global_vector_size,
                     glove_file_name, glove_dimension,
                     local_VSM_dimension, local_VSM_model,
                     arff_file_name, *term_lists_and_their_class_names):
    #this method creates an arff format file that contains local_global vectors for
    #each term in the term list it receives, it also assigns the class_name in the arff vector
    #for each local_global vector
    # kwargs contains the termlists and their respective to class right after each term list:
    #that is, term_list1, its_class, term_list2, its_class etc.
    number_of_term_lists = len(term_lists_and_their_class_names)/2
    class_names = [x for x in term_lists_and_their_class_names
                   if term_lists_and_their_class_names.index(x)%2>0]
    with codecs.open(arff_file_name, "a","utf-8") as arff_file:
        arff_file.write("@RELATION termClass\n\n")
        arff_file.write("@ATTRIBUTE term STRING\n")
        for i in range(local_global_vector_size):
            arff_file.write("@ATTRIBUTE f"+str(i)+" NUMERIC\n")
        #@ATTRIBUTE class        {Iris-setosa,Iris-versicolor,Iris-virginica}
        arff_file.write(r"@ATTRIBUTE class {")
        for i in range(len(class_names)-1):
            arff_file.write(class_names[i]+",")
        arff_file.write(class_names[len(class_names)-1]+"}\n")
        arff_file.write("\n\n@DATA\n")

        for i in range(number_of_term_lists):
            term_list_index = i*2
            term_list_class_name_index = term_list_index+1
            term_list = term_lists_and_their_class_names[term_list_index]
            class_name = term_lists_and_their_class_names[term_list_class_name_index]
            for term in term_list:
                vector = word2localGlobalEmbeddings(term,glove_file_name,glove_dimension,
                                                    local_VSM_dimension,local_VSM_model)
                arff_file.write(term+",")
                for number in vector:
                    arff_file.write(str(number)+",")
                arff_file.write(class_name+"\n")

def subtract_training(dataset_to_subtract_trainin_from, training_set):
    resulting_dataset = [[],[]]
    for i in range(len(dataset_to_subtract_trainin_from[1])):
            if (dataset_to_subtract_trainin_from[0][i] not in training_set[0]):
                #adding such items ro the results
                resulting_dataset[0].append(dataset_to_subtract_trainin_from[0][i])
                resulting_dataset[1].append(dataset_to_subtract_trainin_from[1][i])
    resulting_dataset = [np.array(resulting_dataset[0], np.float32), np.array(resulting_dataset[1], np.int32)]
    return resulting_dataset

def filter_terms_from_tool(trained_classifier, out_put_dataset_to_be_filtered):
    preds = [trained_classifier.predict(datapoint) for datapoint in out_put_to_be_filtered[0]]
    datapoints = list(out_put_dataset_to_be_filtered[0])
    classes = list(out_put_dataset_to_be_filtered[1])
    filtered_datapoints=[]
    filtered_classes=[]
    for i in range(len(preds)):
        if (preds[i]==1):   #this will filter out what the classifier thinks is a non-term.
            filtered_datapoints.append(datapoints[i])
            filtered_classes.append(classes[i])
    #Now build the filtered tool dataset again in np.array format and return it
    filtered_output_dataset = [np.array(filtered_datapoints, np.float32), np.array(filtered_classes, np.int32)]
    #Note: these tool datasets are the outputs of the tools with the correct/wrong classes specified (from annotation)
    return filtered_output_dataset

def word_list_2_dataset(word_list, word_to_glove_vec_dict,
                        glove_dim, local_model, local_model_dim, correct_term_list):
    datapoints = []
    data_classes = []
    util=Utility.Utility()
    for word in word_list:
        datapoint = util.word2localGlobalEmbeddings_models_loaded(word, word_to_glove_vec_dict, glove_dim, model, model_dim)
        the_class = object
        if word in correct_term_list:
            the_class = 1
        else:
            the_class = 0
        datapoints.append(np.array(datapoint, np.float32))
        data_classes.append([the_class])
    data_classes=np.array(data_classes, np.int32)
    data_set=[np.array(datapoints,np.float32), np.array(data_classes,np.float32)]
    return data_set

def get_array_with_class(dataset_array2d, desired_class=[1]):
    #Helper for evaluate_tool
    datapoints_with_desired_class = []
    for i in range(len(dataset_array2d[1])):
        if dataset_array2d[1][i]==desired_class:
            datapoints_with_desired_class.append(dataset_array2d[0][i])
    return datapoints_with_desired_class


def train_and_test(classifier_name,percentage, local_vec_generator, global_vec_generator, collection, data_dir, contextual=False):

    data_set = pickle.load( open( os.path.join(data_dir,"the_dataset.pkl"), "rb" ) ) #CBOW
    data_set_reshaped = pickle.load( open( os.path.join(data_dir,"the_dataset_reshaped.pkl"), "rb" ))

    skfold=sklearn.model_selection.StratifiedKFold(n_splits=10,shuffle=True, random_state=2)
    sfolds_generator=skfold.split(data_set_reshaped[0], data_set_reshaped[1]) #da

    X = data_set[0]
    Y = data_set[1]

    accuracy_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    layer_capacity = 30
    MLP_layers=(layer_capacity, layer_capacity, layer_capacity, layer_capacity)

    classifier_dict = {
        "svm" : sklearn.svm.SVC(kernel='linear'),
        "mlp": sklearn.neural_network.MLPClassifier(MLP_layers),
        "logistic": sklearn.linear_model.LogisticRegression()
    }

    for train_indices, test_indices in sfolds_generator:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = Y[train_indices], Y[test_indices]

        classifier = classifier_dict[classifier_name]
        # print(X_train.shape)
        # input()
        classifier.fit(X_train, y_train)

        preds = [int(i) for i in classifier.predict(X_test)]
        target = [int(i) for i in y_test]

        acc=sklearn.metrics.accuracy_score(target, preds)
        f1=sklearn.metrics.f1_score(target, preds) #ALT_1_A
        precision=sklearn.metrics.precision_score(target, preds) #ALT_1_A
        recall = sklearn.metrics.recall_score(target, preds) #ALT_1_A

        accuracy_list.append(acc)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)

    accuracy_mean = np.mean(accuracy_list)
    f1_mean = np.mean(f1_list)
    precision_mean = np.mean(precision_list)
    recall_mean = np.mean(recall_list)

    new_result = {
        "classifier": classifier_name,
        "local_vec": local_vec_generator,
        "global_vec":global_vec_generator,
        "accuracy": accuracy_mean,
        "f1": f1_mean,
        "precision": precision_mean,
        "recall": recall_mean
    }
    # print(new_result)
    collection.insert_one(new_result)
    print(f"Result for {classifier_name}, {percentage} saved")
    return

def train_and_test_visualize(classifier_name,percentage, local_vec_generator, global_vec_generator, collection, data_dir):

    data_set = pickle.load( open( os.path.join(data_dir,"the_dataset.pkl"), "rb" ) ) #CBOW
    data_set_reshaped = pickle.load( open( os.path.join(data_dir,"the_dataset_reshaped.pkl"), "rb" ))

    skfold=sklearn.model_selection.StratifiedKFold(n_splits=10,shuffle=True, random_state=2)
    sfolds_generator=skfold.split(data_set_reshaped[0], data_set_reshaped[1]) #da

    X = data_set[0]
    Y = data_set[1]
    words = np.array(data_set[2])

    accuracy_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    layer_capacity = 30
    MLP_layers=(layer_capacity, layer_capacity, layer_capacity, layer_capacity)

    classifier_dict = {
        "svm" : sklearn.svm.SVC(kernel='linear'),
        "mlp": sklearn.neural_network.MLPClassifier(MLP_layers),
        "logistic": sklearn.linear_model.LogisticRegression()
    }

    fn_list = []
    fp_list = []
    tp_list = []
    tn_list = []
    classifier_name= "mlp"

    for train_indices, test_indices in sfolds_generator:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = Y[train_indices], Y[test_indices]
        words_train, words_test = words[train_indices], words[test_indices]
        classifier = classifier_dict[classifier_name]

        classifier.fit(X_train, y_train)

        preds = [int(i) for i in classifier.predict(X_test)]
        target = [int(i) for i in y_test]

        for index, i in enumerate(preds):
            if i != target[index]:
                if i==1:
                    fp_list.append(words_test[index])
                else:
                    fn_list.append(words_test[index])
            else:
                if i==1:
                    tp_list.append(words_test[index])
                else:
                    tn_list.append(words_test[index])


    print(f"TP: \n {list(set(tp_list))}\n")
    print(f"FP: \n {list(set(fp_list))}\n")
    print(f"FN: \n {list(set(fn_list))}\n")
    print(f"TN: \n {list(set(tn_list))}\n")

    accuracy_mean = np.mean(accuracy_list)
    f1_mean = np.mean(f1_list)
    precision_mean = np.mean(precision_list)
    recall_mean = np.mean(recall_list)

    new_result = {
        "classifier": classifier_name,
        "local_vec": local_vec_generator,
        "global_vec":global_vec_generator,
        "accuracy": accuracy_mean,
        "f1": f1_mean,
        "precision": precision_mean,
        "recall": recall_mean,
        "contextual" : contextual
    }

    collection.insert_one(new_result)
    print(f"Result for {classifier}, {percentage} saved")
    return

def evaluate_tool_new(evaluator_terms, correct_terms, nonterms):
    tp = 0
    fp = 0
    fn = 0

    for term in evaluator_terms:
        if term in correct_terms:
            tp +=1
        elif term in nonterms:
            fp +=1
    for term in correct_terms:
        if term not in evaluator_terms:
            fn +=1

    precision = float(tp)/(float(tp+fp))
    recall = float(tp)/(float(tp+fn))
    f1_score = 2*(float(precision*recall))/float(precision+recall)

    classifier_name = "TermoStat"

    new_result = {
        "classifier": classifier_name,
        "local_vec": None,
        "global_vec":None,
        "accuracy": None,
        "f1": f1_score,
        "precision": precision,
        "recall": recall
    }

    return new_result

def train_and_test_no_split(classifier_name, local_vec_generator, global_vec_generator, collection, data_dir, contextual=False):

    train_set = pickle.load( open( os.path.join(data_dir,"train.pkl"), "rb" ) ) #CBOW
    eval_set = pickle.load( open( os.path.join(data_dir,"eval.pkl"), "rb" ) ) #CBOW

    X_train= train_set[0]
    y_train = train_set[1]

    X_test = eval_set[0]
    y_test = eval_set[1]


    print(X_train.shape)
    n_samples, nx, ny = X_train.shape

    if contextual:
        X_train = np.reshape(X_train,(n_samples, nx*ny))

    n_samples, nx, ny = X_test.shape

    if contextual:
        X_test = np.reshape(X_test,(n_samples, nx*ny))

    layer_capacity = 30
    MLP_layers=(layer_capacity, layer_capacity, layer_capacity, layer_capacity)

    classifier_dict = {
        "svm" : sklearn.svm.SVC(kernel='linear'),
        "mlp": sklearn.neural_network.MLPClassifier(MLP_layers, verbose=True),
        "logistic": sklearn.linear_model.LogisticRegression()
    }

    classifier = classifier_dict[classifier_name]
    # print(X_train.shape)
    # input()
    classifier.fit(X_train, y_train)

    
    preds = [int(i) for i in classifier.predict(X_test)]
    target = [int(i) for i in y_test]

    acc=sklearn.metrics.accuracy_score(target, preds)
    f1=sklearn.metrics.f1_score(target, preds) #ALT_1_A
    precision=sklearn.metrics.precision_score(target, preds) #ALT_1_A
    recall = sklearn.metrics.recall_score(target, preds) #ALT_1_A

    new_result = {
        "classifier": classifier_name,
        "local_vec": local_vec_generator,
        "global_vec":global_vec_generator,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    print(new_result)
    collection.insert_one(new_result)
    print(f"Result for {classifier_name} saved")
    return

def concat(local_word, local_context, axis):
    # axis=1 to have 1xn
    x_concat = np.concatenate((local_word, local_context), axis=axis)
    return x_concat

def combine_features(X, word_layer = "last"):
    X = pd.DataFrame.from_records(X)
    # print(X)
    # sys.exit()
    if word_layer == "last":
        try:
            X["local_concat"] = X.apply(lambda x: concat(x["local_word_last"], x["local_context"], 0), axis=1)
        except:
            X["local_concat"] = X.apply(lambda x: concat(x["local_word_lastlayer"], x["local_context"], 0), axis=1)
    elif word_layer == "2ndlast":
        try:
            X["local_concat"] = X.apply(lambda x: concat(x["local_word_2ndlast"], x["local_context"], 0), axis=1)
        except:
            X["local_concat"] = X.apply(lambda x: concat(x["local_word_2ndlastlayer"], x["local_context"], 0), axis=1)

    X["global_concat"] = X.apply(lambda x: concat(x["global_word"], x["global_context"], 0), axis=1)
    X["combined"] = X.apply(lambda x: concat(x["local_concat"], x["global_concat"], 0), axis=1)
    X = list(X["combined"])
    X = np.array(X, dtype=np.float32)
    return X 


def train_and_test_concat(classifier_name, local_vec_generator, global_vec_generator, collection, data_dir, window_size):

    train_set = pickle.load( open( os.path.join(data_dir,"train.pkl"), "rb" ) ) #CBOW
    eval_set = pickle.load( open( os.path.join(data_dir,"eval.pkl"), "rb" ) ) #CBOW

    X_train= train_set["features"]
    y_train = train_set["labels"]

    X_test = eval_set["features"]
    y_test = eval_set["labels"]

    word_layer = "last"

    X_train_combined = combine_features(X_train, word_layer=word_layer)
    X_test_combined = combine_features(X_test, word_layer=word_layer) 

    layer_capacity = 30
    MLP_layers=(layer_capacity, layer_capacity, layer_capacity, layer_capacity)

    classifier_dict = {
        "svm" : sklearn.svm.SVC(kernel='linear'),
        "mlp": sklearn.neural_network.MLPClassifier(MLP_layers, verbose=True),
        "logistic": sklearn.linear_model.LogisticRegression()
        # tol = 1e-8, max_iter = 50

    }

    classifier = classifier_dict[classifier_name]
    classifier.fit(X_train_combined, y_train)

    
    preds = [int(i) for i in classifier.predict(X_test_combined)]
    target = [int(i) for i in y_test]

    acc=sklearn.metrics.accuracy_score(target, preds)
    f1=sklearn.metrics.f1_score(target, preds) #ALT_1_A
    precision=sklearn.metrics.precision_score(target, preds) #ALT_1_A
    recall = sklearn.metrics.recall_score(target, preds) #ALT_1_A

    if classifier_name == "mlp":
        iterations = classifier_dict[classifier_name].n_iter_

    new_result = {
        "classifier": classifier_name,
        "local_vec": local_vec_generator,
        "global_vec":global_vec_generator,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "method": "concat",
        "word_layer": word_layer,
        "window_size": window_size, 
        "n_iter": iterations,
    }
    print(new_result)
    # sys.exit()
    collection.insert_one(new_result)
    print(f"Result for {classifier_name} saved")
    return

def combine_tads(X, word_layer = "last"):
    X = pd.DataFrame.from_records(X)
    # print(X)
    # sys.exit()
    if word_layer == "last":
        try:
            X["local_concat"] = X.apply(lambda x: concat(x["local_word_last"], x["local_context"], 0), axis=1)
        except:
            X["local_concat"] = X.apply(lambda x: concat(x["local_word_lastlayer"], x["local_context"], 0), axis=1)
    elif word_layer == "2ndlast":
        try:
            X["local_concat"] = X.apply(lambda x: concat(x["local_word_2ndlast"], x["local_context"], 0), axis=1)
        except:
            X["local_concat"] = X.apply(lambda x: concat(x["local_word_2ndlastlayer"], x["local_context"], 0), axis=1)

    X = list(X["local_concat"])
    X = np.array(X, dtype=np.float32)
    return X 

def train_and_test_concat_tads(classifier_name, local_vec_generator, global_vec_generator, collection, data_dir, window_size):

    train_set = pickle.load( open( os.path.join(data_dir,"train.pkl"), "rb" ) ) #CBOW
    eval_set = pickle.load( open( os.path.join(data_dir,"eval.pkl"), "rb" ) ) #CBOW

    X_train= train_set["features"]
    y_train = train_set["labels"]

    X_test = eval_set["features"]
    y_test = eval_set["labels"]

    word_layer = "last"

    X_train_combined = combine_tads(X_train, word_layer=word_layer)
    X_test_combined = combine_tads(X_test, word_layer=word_layer) 

    layer_capacity = 30
    MLP_layers=(layer_capacity, layer_capacity, layer_capacity, layer_capacity)

    classifier_dict = {
        "svm" : sklearn.svm.SVC(kernel='linear'),
        "mlp": sklearn.neural_network.MLPClassifier(MLP_layers, verbose=True),
        "logistic": sklearn.linear_model.LogisticRegression()
        # tol = 1e-8, max_iter = 50

    }

    classifier = classifier_dict[classifier_name]
    classifier.fit(X_train_combined, y_train)

    
    preds = [int(i) for i in classifier.predict(X_test_combined)]
    target = [int(i) for i in y_test]

    acc=sklearn.metrics.accuracy_score(target, preds)
    f1=sklearn.metrics.f1_score(target, preds) #ALT_1_A
    precision=sklearn.metrics.precision_score(target, preds) #ALT_1_A
    recall = sklearn.metrics.recall_score(target, preds) #ALT_1_A

    if classifier_name == "mlp":
        iterations = classifier_dict[classifier_name].n_iter_

    new_result = {
        "classifier": classifier_name,
        "local_vec": local_vec_generator,
        "global_vec":global_vec_generator,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "method": "concat-local-only",
        "word_layer": word_layer,
        "window_size": window_size, 
        "n_iter": iterations,
    }
    print(new_result)
    # sys.exit()
    collection.insert_one(new_result)
    print(f"Result for {classifier_name} saved")
    return

def get_context_column(X):
    X = pd.DataFrame.from_records(X)
    X["context"] = X.apply(lambda x: concat(x["local_context"], x["global_context"], 0), axis=1)
    X = list(X["context"])
    X = np.array(X, dtype=np.float32)
    return X 

def train_and_test_context(classifier_name, local_vec_generator, global_vec_generator, collection, data_dir, window_size):

    train_set = pickle.load( open( os.path.join(data_dir,"train.pkl"), "rb" ) ) #CBOW
    eval_set = pickle.load( open( os.path.join(data_dir,"eval.pkl"), "rb" ) ) #CBOW

    X_train= train_set["features"]
    y_train = train_set["labels"]

    X_test = eval_set["features"]
    y_test = eval_set["labels"]

    word_layer = "last"

    X_train_context = get_context_column(X_train)
    X_test_context = get_context_column(X_test)

    # X_test_combined = combine_features(X_test, word_layer=word_layer) 

    layer_capacity = 30
    MLP_layers=(layer_capacity, layer_capacity, layer_capacity, layer_capacity)

    classifier_dict = {
        "svm" : sklearn.svm.SVC(kernel='linear'),
        "mlp": sklearn.neural_network.MLPClassifier(MLP_layers, verbose=True, max_iter = 15),
        "logistic": sklearn.linear_model.LogisticRegression()
        # tol = 1e-8, max_iter = 50
    }

    classifier = classifier_dict[classifier_name]
    classifier.fit(X_train_context, y_train)

    
    preds = [int(i) for i in classifier.predict(X_test_context)]
    target = [int(i) for i in y_test]

    acc=sklearn.metrics.accuracy_score(target, preds)
    f1=sklearn.metrics.f1_score(target, preds) #ALT_1_A
    precision=sklearn.metrics.precision_score(target, preds) #ALT_1_A
    recall = sklearn.metrics.recall_score(target, preds) #ALT_1_A

    if classifier_name == "mlp":
        iterations = classifier_dict[classifier_name].n_iter_

    new_result = {
        "classifier": classifier_name,
        "local_vec": local_vec_generator,
        "global_vec":global_vec_generator,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "method": "context_only",
        "word_layer": word_layer,
        "window_size": window_size, 
        "n_iter": iterations,
    }
    print(new_result)
    # sys.exit()
    collection.insert_one(new_result)
    print(f"Result for {classifier_name} saved")
    return

def train_and_test_benchmark(classifier_name, local_vec_generator, global_vec_generator, collection, data_dir, contextual=False):

    train_set = pickle.load( open( os.path.join(data_dir,"train.pkl"), "rb" ) ) #CBOW
    eval_set = pickle.load( open( os.path.join(data_dir,"eval.pkl"), "rb" ) ) #CBOW

    X_train= train_set["features"]
    y_train = train_set["labels"]

    X_test = eval_set["features"]
    y_test = eval_set["labels"]

    word_layer = "last"
    X_train = pd.DataFrame.from_records(X_train)
    X_test = pd.DataFrame.from_records(X_test)
    try:
        X_train_word = np.array(list(X_train[f"local_word_{word_layer}layer"]), dtype=np.float32)
        X_test_word = np.array(list(X_test[f"local_word_{word_layer}layer"]), dtype=np.float32)
    except:
        X_train_word = np.array(list(X_train[f"local_word_{word_layer}"]), dtype=np.float32)
        X_test_word = np.array(list(X_test[f"local_word_{word_layer}"]), dtype=np.float32)

    # X_test_combined = combine_features(X_test) 

    layer_capacity = 30

    # MLP_layers=(120, 80, 60, 30)
    # MLP_layers=(120, 80, 60, 30, 10, 5)
    MLP_layers=(layer_capacity, layer_capacity, layer_capacity, layer_capacity)

    classifier_dict = {
        "svm" : sklearn.svm.SVC(kernel='linear'),
        "mlp": sklearn.neural_network.MLPClassifier(MLP_layers, verbose=True),
        "logistic": sklearn.linear_model.LogisticRegression()
    }

    classifier = classifier_dict[classifier_name]
    classifier.fit(X_train_word, y_train)

    
    preds = [int(i) for i in classifier.predict(X_test_word)]
    target = [int(i) for i in y_test]

    acc=sklearn.metrics.accuracy_score(target, preds)
    f1=sklearn.metrics.f1_score(target, preds) #ALT_1_A
    precision=sklearn.metrics.precision_score(target, preds) #ALT_1_A
    recall = sklearn.metrics.recall_score(target, preds) #ALT_1_A

    if classifier_name == "mlp":
        iterations = classifier_dict[classifier_name].n_iter_
    new_result = {
        "classifier": classifier_name,
        "local_vec": local_vec_generator,
        "global_vec":global_vec_generator,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "method": "benchmark",
        "word_layer": word_layer,
        "n_iter": iterations,
    }
    print(new_result)
    # sys.exit()
    collection.insert_one(new_result)
    print(f"Result for {classifier_name} saved")
    return

def split_target_context(X, word_layer = "last"):
    X = pd.DataFrame.from_records(X)
    try:
        X_target = np.array(list(X[f"local_word_{word_layer}"]), dtype=np.float32)
    except:
        X_target = np.array(list(X[f"local_word_{word_layer}layer"]), dtype=np.float32)

    X_context = np.array(list(X["local_context"]), dtype=np.float32)
    
    return X_target, X_context 

def split_local_global(X, word_layer = "last"):
    X = pd.DataFrame.from_records(X)
    try:
        X_local = np.array(list(X[f"local_word_{word_layer}"]), dtype=np.float32)
    except:
        X_local = np.array(list(X[f"local_word_{word_layer}layer"]), dtype=np.float32)

    X_global = np.array(list(X["global_word"]), dtype=np.float32)

    return X_local, X_global 


def train_and_test_siamese(classifier_name, local_vec_generator, global_vec_generator, collection, data_dir, window_size, vanilla=False):

    train_set = pickle.load( open( os.path.join(data_dir,"train.pkl"), "rb" ) ) #CBOW
    eval_set = pickle.load( open( os.path.join(data_dir,"eval.pkl"), "rb" ) ) #CBOW

    X_train= train_set["features"]
    y_train = train_set["labels"]

    X_test = eval_set["features"]
    y_test = eval_set["labels"]

    word_layer = "last"

    if vanilla:
        X_train_target, X_train_context = split_target_context(X_train, word_layer = word_layer)
        X_test_target, X_test_context = split_target_context(X_test, word_layer = word_layer)

    else:
        X_train_target, X_train_context = split_local_global(X_train, word_layer = word_layer)
        X_test_target, X_test_context = split_local_global(X_test, word_layer = word_layer)

    X_train_target, X_train_context, y_train = shuffle(X_train_target, X_train_context, y_train)
    X_test_target, X_test_context, y_test = shuffle(X_test_target, X_test_context, y_test)

    x_target_shape = X_train_target.shape
    x_context_shape = X_train_context.shape
    
    model = keras.Sequential([
        keras.Input(shape=(x_target_shape[1])),
        layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
        layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
        layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
        layers.Dense(30, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
    ])

    model2 = keras.Sequential([
        keras.Input(shape=(x_target_shape[1])),
        layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
        layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
        layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
        layers.Dense(30, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
    ])


    siamese_channel = keras.Sequential([
        keras.Input(shape=(x_target_shape[1])),
        layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
        layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
        layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
        layers.Dense(30, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)),
    ])

    # model.add()
    left_input = keras.Input(x_target_shape[1])
    right_input = keras.Input(x_context_shape[1])

    encoded_l = model(left_input)
    encoded_r = model2(right_input)

    siamese_encoded = siamese_channel(left_input)
    siamese_encoded_r = siamese_channel(right_input)
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = layers.Lambda(lambda tensors:keras.backend.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([siamese_encoded, siamese_encoded_r])
    
    concatenated = layers.Concatenate(axis=1)([L1_distance, encoded_l, encoded_r])
    # concatenated = layers.Dot(axes=1)([L1_distance, encoded_l, encoded_r])
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction= layers.Dense(1, activation="sigmoid")(concatenated)
    # shrink = layers.Dense(1)(concatenated)
    # prediction = layers.Activation('softmax')(shrink)

    siamese_model = keras.Model(inputs=[left_input, right_input], outputs=prediction)
    print(siamese_model.summary())
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(),
                Precision(), Recall()])

    print(y_train.shape)
    history = siamese_model.fit(
        [X_train_target, X_train_context],
        y_train,
        batch_size=64,
        epochs=5,
    )
    
    results = siamese_model.evaluate([X_test_target, X_test_context], y_test, batch_size=128)
    print(siamese_model.metrics_names)
    print(results)

    acc = results[1]
    precision = results[2]
    recall = results[3]

    f1 = 2*((precision*recall)/(precision+recall))

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

    new_result = {
        "classifier": classifier_name,
        "local_vec": local_vec_generator,
        "global_vec":global_vec_generator,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "method": "multi-lgv",
        "word_layer": word_layer,
        "window_size": window_size
    }
    print(new_result)
    collection.insert_one(new_result)
    print(f"Result for {classifier_name} saved")
    return