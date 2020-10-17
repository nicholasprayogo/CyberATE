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
    Multichannel
)




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

def train_multichannel(local_vec_generator, global_vec_generator, collection, data_dir, n_epochs, batch_size, hidden_dim):

    shuffle = False
    num_workers = 4

    classifier_name = "multichannel"

    data_set_local = pickle.load( open( os.path.join(data_dir,"the_dataset_local.pkl"), "rb" ) ) #CBOW
    data_set_local_reshaped = pickle.load( open( os.path.join(data_dir,"the_dataset_local_reshaped.pkl"), "rb" ))
    data_set_global = pickle.load( open( os.path.join(data_dir,"the_dataset_global.pkl"), "rb" ) ) #CBOW
    data_set_global_reshaped = pickle.load( open( os.path.join(data_dir,"the_dataset_global_reshaped.pkl"), "rb" ))

    X_local = data_set_local[0]
    X_global = data_set_global[0]
    Y = data_set_global[1]

    accuracy_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    # hyperparam search

    learning_rate_list = list(np.linspace(0.001, 0.3, 10))
    momentum_list = list(np.linspace(0.0, 0.9, 10))
    activations=["relu", "tanh"]

    torch.manual_seed(32)

    skfold=sklearn.model_selection.StratifiedKFold(n_splits=10,shuffle=True, random_state=2)
    sfolds_generator=skfold.split(data_set_local[0], data_set_local[1]) #da
    momentum = 0.9
    learning_rate = 0.07
    activation = "relu"

    for train_indices, test_indices in tqdm(sfolds_generator):
        X_train_local, X_test_local = X_local[train_indices], X_local[test_indices]
        X_train_global, X_test_global = X_global[train_indices], X_global[test_indices]
        y_train, y_test = Y[train_indices], Y[test_indices]

        torch_training_set = CustomDataset(X_train_local, X_train_global, y_train)
        train_loader = DataLoader(torch_training_set, shuffle=True, batch_size=batch_size)

        x_local_shape = X_train_local.shape
        x_global_shape = X_train_global.shape
        y_local_shape = y_train.shape
        y_global_shape = y_train.shape
        y_label_shape= y_global_shape

        local_vecs = np.transpose(X_train_local)
        global_vecs = np.transpose(X_train_global)
        local_shape = local_vecs.shape
        global_shape = global_vecs.shape


        model = Multichannel(batch_size, x_local_shape, x_global_shape, y_label_shape, hidden_dim, activation)

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        model.train()

        for epoch in range(n_epochs):
            running_loss =0.0
            for batch_index, batch in enumerate(train_loader):

                x_local, x_global, y = batch
                optimizer.zero_grad()

                y = torch.flatten(y)
                y_pred = model(x_local, x_global)

                y = torch.tensor([float(i) for i in y], dtype=torch.float32)

                y_pred = torch.tensor([float(i) for i in y_pred.flatten()], dtype= torch.float32, requires_grad=True)

                criterion = nn.BCELoss()
                loss = criterion(y_pred, y)

                running_loss += loss.item()
                if batch_index==13:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, batch_index + 1, running_loss / 13))
                    running_loss = 0.0
                    print(f"Loss: {loss.item()}")
                loss.backward()
                optimizer.step()

        preds = model(Variable(torch.from_numpy(X_test_local)), Variable(torch.from_numpy(X_test_global)))

        preds_binary = [1 if pred<0.5 else 0 for pred in preds]
        target = y_test.flatten()

        acc=sklearn.metrics.accuracy_score(target, preds_binary)
        f1=sklearn.metrics.f1_score(target, preds_binary, average="macro") #ALT_1_A
        precision=sklearn.metrics.precision_score(target, preds_binary, average="macro") #ALT_1_A
        recall = sklearn.metrics.recall_score(target, preds_binary, average="macro") #ALT_1_A

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
        "n_epochs": n_epochs,
        "local_vec": local_vec_generator,
        "global_vec":global_vec_generator,
        "accuracy": accuracy_mean,
        "f1": f1_mean,
        "precision": precision_mean,
        "recall": recall_mean,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "activation": activation
    }

    collection.insert_one(new_result)

    print(f"Results for multichannel saved")
    return

def train_and_test(classifier_name,percentage, local_vec_generator, global_vec_generator, collection, data_dir):

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
        "recall": recall_mean
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

def create_multichannel_data(datapoints_local, datapoints_global, data_classes):

    df = pd.DataFrame({"local":datapoints_local, "global":datapoints_global, "Y":list(data_classes)})

    df_minority = df[df["Y"]==1]
    df_majority = df[df["Y"]==0]

    df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=int(df_majority.size/3),
                                 random_state=2)

    df_upsampled = pd.concat([df_minority_upsampled,df_majority]).reset_index()

    datapoints_local = df_upsampled["local"]
    datapoints_global = df_upsampled["global"]
    data_classes = df_upsampled["Y"]

    new_datapoints_local =[]
    new_datapoints_global =[]
    new_data_classes = []

    for i in datapoints_local:
        new_datapoints_local.append(list(i))

    for i in datapoints_global:
        new_datapoints_global.append(list(i))

    for i in data_classes:
        new_data_classes.append(int(i))

    datapoints_local = np.array(new_datapoints_local)
    datapoints_global = np.array(new_datapoints_global)
    data_classes = np.array(new_data_classes)

    return datapoints_local, datapoints_global, data_classes
