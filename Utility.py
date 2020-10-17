import logging
import codecs
import random
import nltk
from nltk import *
from collections import defaultdict
from gensim import corpora, models, similarities
import copy
from copy import *
import scipy
import numpy as np
import sys
import os
import time
from time import gmtime, strftime, localtime
import torch

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Utility(object):

    tiny_float = 0.0000000001

    def corpus2wordVSM(self, corpus_file_name, embeddings_file_name=r"embeddings.w2v",
                   feature_vec_size=100, window_size=5, minimum_count=5, num_of_virtual_cores=4, skipGram=0):
        corpus_file = codecs.open(corpus_file_name, "r", "utf-8")
        corpus_text = corpus_file.read().lower()
        corpus_file.close()
        corpus_sentence_tokenized = nltk.tokenize.sent_tokenize(corpus_text)

        #making sure the file is new when created below
        if os.path.isfile("sent_line_corpus.txt"):
            os.remove(r"sent_line_corpus.txt")
        sent_line_corpus_file = codecs.open(r"sent_line_corpus.txt", "a", "utf-8")
        for sentence in corpus_sentence_tokenized:
            sent_line_corpus_file.write("\n")
            word_tokens = nltk.tokenize.word_tokenize(sentence)
            for word in word_tokens:
                # if isinstance(word, unicode): #deprecated
                sent_line_corpus_file.write(word.lower()+" ")
        sent_line_corpus_file.close()
        sentences = models.word2vec.LineSentence(r"sent_line_corpus.txt")
        model = models.Word2Vec(sentences, size=feature_vec_size, window=5, min_count=5, workers=4, sg=skipGram)
        #if the parameter sg=0 (defult) is changed to sg=1,
        #the model will be skip-gram as opposed to CBOW
        model.save(embeddings_file_name)

    def corpus2wordVSM_CBOW(self,corpus_file_name, embeddings_file_name=r"embeddings.w2v",
                       feature_vec_size=100, window_size=5, minimum_count=5, num_of_virtual_cores=4):
        math_corpus_file = codecs.open(corpus_file_name, "r", "utf-8")
        math_corpus_text = math_corpus_file.read().lower()
        math_corpus_file.close()
        math_corpus_sanitized_sentence_tokenized = tokenize.sent_tokenize(math_corpus_text)
        sent_line_math_corpus_file = open(r"sent_line_math_corpus.txt", "a")
        for sentence in math_corpus_sanitized_sentence_tokenized:
            sent_line_math_corpus_file.write("\n")
            word_tokens = tokenize.word_tokenize(sentence)
            for word in word_tokens:
                # if isinstance(word, unicode):
                sent_line_math_corpus_file.write(word.lower()+" ")
        sent_line_math_corpus_file.close()
        sentences = models.word2vec.LineSentence(r"sent_line_math_corpus.txt")
        model = models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
        model.save(embeddings_file_name)

    def terms2list(self, terms_file_name):
        terms_file = codecs.open(terms_file_name, "r", "utf-8")
        term_list = terms_file.read().splitlines()
        terms_file.close()
        return term_list

    def single_terms_clean_up(self, word_list):
        cleaned_single_terms = [item for item in word_list
                         if ((len(item.split())==1) and
                             ("-" not in item) and ("|" not in item))]
        return cleaned_single_terms

    def word_list_to_glove_vectors_dict(self, word_list, embeddings_file_name):
        #This function returns a dictionary containing the words (in the list) as keys and their corresponding
        #vectors (in the embedding file streamed by a line generator, GloVe for this case) as values
        #This function returns the values (that are vectors) as numpy arrays

        embeddings_list = [] #each embedding is a list/vector so this is a nested list.
                             #each vector has the word as its first item
        vectors_dict = dict()
        with open (embeddings_file_name) as embeddings_file:
            #first creating a generator to stream the big file's data
            for embedding_string in embeddings_file:#this is a generator that streams lines one by one,
                #so each embedding_string is a line in embeddings_file
                for word in word_list:
                    embedding_vector = embedding_string.split()
                    if (word == embedding_vector[0]):
                        embeddings_list.append(embedding_vector)
            #now building a dictionary for words as keys and their corresponding vectors as values
        for embedding in embeddings_list:
            key = embedding[0]
            value = [float(x) for x in embedding[1:]]
            vectors_dict.update({key:np.array(value)})
        return vectors_dict

    def get_glove_vector(self, word, glove_file_name, size_of_not_found_vector=50):
        with codecs.open(glove_file_name, "r", "utf-8") as glove_file:
            vector = []
            returning_vector = []
            for line in glove_file:
                lineList = line.split()
                if lineList[0] == word:
                    vector = lineList[1:]
            if len(vector)!=0:
                returning_vector = vector
            else: returning_vector = [self.tiny_float]*size_of_not_found_vector #not 0 to avoid "division by 0 error later on when computing assessment scores"
            #code for random vector instead of zeros vector:
                #RandomVector = [random.randint(LowerBound,UpperBound) for i in range(VectorSize)]
        return returning_vector

    def get_glove_global_vector_from_model(self, word, glove_dict, size_of_not_found_vector=50):
        vector = []
        if (word in glove_dict.keys()):
            returning_vector = [float(i) for i in list(np.array(glove_dict[word]).flatten())]
        else: returning_vector = [float(i) for i in [self.tiny_float]*size_of_not_found_vector] #not 0 to avoid "division by 0 error later on when computing assessment scores"
        #code for random vector instead of zeros vector:
            #RandomVector = [random.randint(LowerBound,UpperBound) for i in range(VectorSize)]

        return returning_vector


    def word2localGlobalEmbeddings_models_loaded(self, word, glove_dict, glove_dimension,
                                   local_model, local_dimension):
        global_vec = self.get_glove_global_vector_from_model(word, glove_dict, glove_dimension)
        local_vec = self.w2v_term2local_vector(word, local_dimension, local_model)

        local_global_embeddings = local_vec+global_vec

        return local_global_embeddings

    def get_bert_global_vector_from_model(self, word, bert_dict, size_of_not_found_vector=768):
        vector = []

        if (word in bert_dict.keys()):
            returning_vector = [float(i) for i in list(np.array(bert_dict[word]).flatten())] #TODO Fix on bert embedding generation

        else: returning_vector = [self.tiny_float]*size_of_not_found_vector #not 0 to avoid "division by 0 error later on when computing assessment scores"
        #code for random vector instead of zeros vector:
            #RandomVector = [random.randint(LowerBound,UpperBound) for i in range(VectorSize)]

        return returning_vector

    def word2localGlobalEmbeddings_models_separated(self, word, glove_dict, glove_dimension,
                                   local_model, local_dimension):
        global_vec = self.get_glove_global_vector_from_model(word, glove_dict, glove_dimension)
        local_vec = self.w2v_term2local_vector(word, local_dimension, local_model)

        return global_vec, local_vec


    def new_word2localGlobalEmbeddings_models_localonly(self, word, global_model_type, global_dict, global_dimension,
                                   local_model_type, local_model, local_dimension, local_tokenizer=None):

        if local_model_type == "w2v":
            local_vec = self.w2v_term2local_vector(word, local_dimension, local_model)
        elif local_model_type in ["bert", "bert_raw", "bert_finetune", "bert_refined"]:
            local_vec = self.bert_term2local_vector(word, local_dimension, local_model, local_tokenizer)

        return local_vec

    def new_word2localGlobalEmbeddings_models_separated(self, word, global_model_type, global_dict, global_dimension,
                                   local_model_type, local_model, local_dimension, local_tokenizer=None):

        if local_model_type == "w2v":
            local_vec = self.w2v_term2local_vector(word, local_dimension, local_model)
        elif local_model_type in ["bert", "bert_raw"]:
            local_vec = self.bert_term2local_vector(word, local_dimension, local_model, local_tokenizer)

        if global_model_type == "glove":
            global_vec = self.get_glove_global_vector_from_model(word, global_dict, global_dimension)
        elif global_model_type == "bert":
            global_vec = self.get_bert_global_vector_from_model(word, global_dict, global_dimension)

        return global_vec, local_vec

    def new_word2localGlobalEmbeddings_models(self, word, global_model_type, global_dict, global_dimension,
                                   local_model_type, local_model, local_dimension, local_tokenizer=None):

        if local_model_type == "w2v":
            local_vec = self.w2v_term2local_vector(word, local_dimension, local_model)
        elif local_model_type in ["bert", "bert_raw", "bert_finetune",  "bert_refined"]:
            local_vec = self.bert_term2local_vector(word, local_dimension, local_model, local_tokenizer)

        if global_model_type == "glove":
            global_vec = self.get_glove_global_vector_from_model(word, global_dict, global_dimension)
        elif global_model_type == "bert":
            global_vec = self.get_bert_global_vector_from_model(word, global_dict, global_dimension)

        local_global_embeddings = local_vec+global_vec

        return local_global_embeddings

    def bert_word2localGlobalEmbeddings(self, word, global_model_type, global_file_name, global_dimension,
                                   local_model_type, local_model, local_dimension):
        global_vec = self.get_bert_global_vector_from_model(word, bert_file_name, bert_dimension)
        local_vec = self.w2v_term2local_vector(word, local_dimension, local_model)
        local_global_embeddings = local_vec+global_vec
        return local_global_embeddings


    def word2localGlobalEmbeddings(self, word, glove_file_name, glove_dimension,
                                   local_model, local_dimension):
        global_vec = self.get_glove_vector(word, glove_file_name, glove_dimension)
        local_vec = self.w2v_term2local_vector(word, local_dimension, local_model)

        local_global_embeddings = local_vec+global_vec
        return local_global_embeddings

    def bert_term2local_vector(self, term, size_of_not_found_vec, local_bert_model, tokenizer):

        input_ids = torch.tensor(tokenizer.encode(term)).unsqueeze(0)
        hidden_states = local_bert_model(input_ids)
        token_embeddings = hidden_states[0].permute(1,0,2) # put tokens dimension in front

        tokens =[]
        for token_index, token in enumerate(token_embeddings):
          if token_index not in [0, len(token_embeddings)-1]:
            tokens.append(token)

        word_vector = sum(tokens)
        word_vector = [float(i) for i in list(word_vector.detach().numpy().flatten())]

        return word_vector

    def w2v_term2local_vector(self, term, size_of_not_found_vector, local_model):
        vector = []
        if local_model.wv.vocab.__contains__(term):
            vector = local_model[term].tolist() #since it's a numpy vector and needs to be
            #turned into a simple list to work for my algorithm

        else: vector = [self.tiny_float]*size_of_not_found_vector

        return vector

    def term_list_to_local_vec_dict(self, term_list, local_model):
        local_vec_dict = {}
        terms_not_in_model = []
        terms_not_in_model
        for word in term_list:
            if local_model.wv.vocab.__contains__(word):
                local_vec_dict[word] = local_model[word].tolist()
            else:
                terms_not_in_model.append(word)
        return local_vec_dict, terms_not_in_model

    def build_local_global_list(self, local_vecs, global_vecs, label):
        words = []
        vectors = []
        labels = []
        # making sure we have both local and global vectors for all the words
        # in other words we're buolding vectors only for words that have both
        # local and global represenation in our data.
        words = list(set(local_vecs.keys()).intersection(set(global_vecs.keys())))
        for word in words:
            vectors.append(local_vecs[word]+global_vecs[word])
            labels.append(label)
        return [words, vectors, labels]

    def term_list_2_ARFF(self, local_global_vector_size,
                         glove_file_name, glove_dimension,
                         local_dimension, local_model,
                         arff_file_name, *term_lists_and_their_class_names):
        #this method creates an arff format file that contains local_global vectors for
        #each term in the term list it receives, it also assigns the class_name in the arff vector
        #for each local_global vector
        # kwargs contains the termlists and their respective class right after each term list:
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
                                                        local_dimension,local_model)
                    arff_file.write(term+",")
                    for number in vector:
                        arff_file.write(str(number)+",")
                    arff_file.write(class_name+"\n")
    def get_local_global_distance(self, word, local_model, size_of_not_found_vector,
                                  global_vectors_dict):
        local_vec = self.w2v_term2local_vector(word,size_of_not_found_vector,local_model)
        global_vec = global_vectors_dict[word]
        distance = scipy.spatial.distance.cosine(local_vec, global_vec)
        return distance
    def classify_by_distance_lower_bound(self, local_model, global_vectors_dict,
                                       lower_bound, upper_bound, terms_list,
                                       non_terms_list, size_of_not_found_vector):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        classified_as_term = []
        classified_as_non_term = []
        for word in terms_list:
            distance = self.get_local_global_distance(word,
                                                      local_model, size_of_not_found_vector,global_vectors_dict)
            if ((abs(distance)>=lower_bound) and
                (abs(distance)<upper_bound)):
                classified_as_term.append(word)
                TP+=1
            else:
                classified_as_non_term.append(word)
                FN+=1
        for word in non_terms_list:
            distance = self.get_local_global_distance(word, local_model, size_of_not_found_vector, global_vectors_dict)
            if ((abs(distance)>=lower_bound) and
                (abs(distance)<upper_bound)):
                classified_as_term.append(word)
                FP+=1
            else:
                classified_as_non_term.append(word)
                TN+=1
        precision = (TP/(float(TP+FP)+self.tiny_float)) #addition of tiny float is
                                                     #to avoid division by 0
        recall = TP/(float(TP+FN)+self.tiny_float)
        f_score = 2*((precision*recall)/(precision+recall+self.tiny_float))
        accuracy = (TP+TN)/(float(TP+TN+FP+FN)+self.tiny_float)
        TPR = TP/(float(TP+FN)+self.tiny_float)
        FPR = FP/(float(FP+TN)+self.tiny_float)
        scores_dict = {"precision":precision, "recall":recall,
                       "f_score":f_score, "accuracy":accuracy,
                       "TPR":TPR, "FPR":FPR, "TP":TP, "FP":FP, "TN":TN, "FN":FN,
                       "classified_as_term":classified_as_term,
                       "classified_as_nonterm":classified_as_non_term}
        return scores_dict


    def granularity_2_lower_and_upper_bounds(self, granularity):
        bounds=[]
        step = (1/float(granularity))
        for x in range(granularity):
            bound = []
            lower_bound = step*x
            upper_bound = (lower_bound+step)
            bound = deepcopy([lower_bound, upper_bound])
            bounds.append(bound)
        return bounds
        #bounds is a list of lists(i.e. pairs) of lower and upper bonds respectively
    def term_list_2_singleWord_term_list(self, term_list):
        return [item for item in term_list
                         if ((len(item.split())==1) and
                             ("-" not in item) and ("|" not in item))]
    def str_glove_vec_2_float_glove_vec(self,str_glove_vec):
        float_glove_vec = [float(item) for item in str_glove_vec]
        return float_glove_vec
    def term_list_2_global_term_vector_dict(self, term_list, glove_file_name, size_of_not_found_vec):
        print("building vector dictionary")
        term_vec_dict = dict()
        for word in term_list:
            term_vec_dict[word]=self.str_glove_vec_2_float_glove_vec(
                self.get_glove_vector(word, glove_file_name, size_of_not_found_vec))
        print("Finished building vector dictionary")
        return term_vec_dict
    def TermoStatOutput2List(self, file_name):
        output_terms_list = []
        with codecs.open(file_name, "r", "utf-8") as file:
            for line in file:
                if line.strip():
                    output_terms_list.append(line.split()[0])
        return output_terms_list
    def words_2_glove_vector_dict(self, list_of_words, glove_dict):
        word_glove_dict = dict()
        words_not_in_glove = []
        for word in list_of_words:
            if word in glove_dict:
                word_glove_dict[word] = glove_dict[word]
            else: words_not_in_glove.append(word)
        return word_glove_dict, words_not_in_glove
    def sample_non_terms(self, sample_size, vocab, terms):
        #np.random.seed(1)
        non_terms = []
        while len(non_terms)<sample_size:
            i = np.random.randint(len(vocab))
            if vocab[i] not in terms:
                non_terms.append(vocab[i])
        return non_terms
