import nltk
import codecs
import numpy as np
from copy import copy, deepcopy
import pickle
import re
import sys
import os
import time
from time import gmtime, strftime, localtime
from tqdm import tqdm

class Preprocess(object):
    """This class prepares the data for the experiments: 1. tokenization 2. casing
    3. preparing the datase (be it training/validation/test sets)"""

    glove_file_name="glove.6B.50d.txt"
    embeddings_dim=50

    def read_lower(self, file_location):
        with codecs.open(file_location, "r", "utf-8") as file:
        #with open(file_location, "r") as file:
        #with open(file_location, "r", encoding="utf-8") as file:
            text = file.read().lower()
            #text = unicode(text)
            text = text.encode('utf-8').strip()
            return text
    def word_tokenize(self, text):
        tokens=nltk.tokenize.word_tokenize(text.decode('utf-8')) #I may want to remove the "decode" function
        #tokens=nltk.tokenize.word_tokenize(text)
        return tokens
    #Build dataset 1 with 20 words before and 5 words after "a/the", the dataset can be a list that holds all the cases, each case is a list where the first
    #item is the datapoint (another list) and the second item is the correct answer "a/the" or 0/1 respectively.
    def a_or_the(self, token=''):
        #Helper
        if (token == 'a' or token == 'the'):
            return True
        else:
            return False
    def innerloop(self, word, glove_file_name, embeddings_list):
        #Helper
            #Loops through the embedding file
            #first creating a generator to stream the big file's data
        with codecs.open (glove_file_name, 'r', 'utf-8') as embeddings_file: #This is to stream (using generators) when the file is large
            for embedding_string in embeddings_file:#this is a generator that streams lines one by one,
                #so each embedding_string is a line in embeddings_file
                embedding_vector = embedding_string.split()
                if (word == embedding_vector[0]):
                    embeddings_list.append(embedding_vector)
                    return embeddings_list
            else:
                return embeddings_list
    def innerloop_small_glove(self, word, glove_file_name, embeddings_list):
        #Helper
            #Loops through the embedding file
        embeddings_file = codecs.open (glove_file_name, 'r', 'utf-8')
        for embedding_string in embeddings_file: #this is a generator that streams lines one by one,
            #so each embedding_string is a line in embeddings_file
            embedding_vector = embedding_string.split()
            if (word == embedding_vector[0]):
                embeddings_list.append(embedding_vector)
                return embeddings_list
        else:
            return embeddings_list

    def vocab_to_dict(self, vocab=[], glove_file_name=glove_file_name, verbose=False):

        vocab_size=len(vocab)
        counter = 0
        slice=vocab_size/100
        check_point=slice
        embeddings_list = [] #each embedding is a list/vector so this is a nested list.
                             #each vector has the word as its first item
        vectors_dict = dict()
        if verbose: print("Started selecting embeddings for the vocabulary")
        #with codecs.open (glove_file_name, 'r', 'utf-8') as embeddings_file: #This is to stream (using generators) when the file is large
        #embeddings_file= codecs.open (glove_file_name, 'r', 'utf-8')
        for word in vocab:
            if (counter>check_point):
                print(str(int((float(counter)/float(vocab_size))*100))+r"%"+" of vocabulary vector dictionary completed")
                check_point+=slice
            embeddings_list = self.innerloop_small_glove(word, glove_file_name, embeddings_list)
            counter+=1
            #now building a dictionary for words as keys and their corresponding vectors as values
        if verbose: print("Finished selecting embeddings for the vocabulary")
        if verbose: print("Started normalizing the digits and creating the vector dictionary")
        for embedding in embeddings_list:
            key = embedding[0]
            value = [np.float32(x) for x in embedding[1:]]
            vectors_dict.update({key:value})
            #vectors_dict.update({key:np.array(value)}) I need them as list for now so I commented this out,
            #I'll convert them to np.arrays later
        if verbose: print("Finished normalizing the digits and created the vector dictionary")
        return vectors_dict
        #works fine so far
    def vocab_to_dict_for_large_vocab(self, vocab=[], glove_file_name=glove_file_name, embedding_file_vocab_size=400000, verbose=True):
        #Helper
        if verbose: print("Started selecting embeddings for the vocabulary")
        embeddings_file = codecs.open(glove_file_name, 'r', 'utf-8')
        file_size=float(embedding_file_vocab_size)
        print("number of lines in the glove embeddings file is:", file_size)
        counter = 0
        slice=file_size/100
        check_point=slice
        embeddings_list = [] #each embedding is a list/vector so this is a nested list.
                             #each vector has the word as its first item
        for embedding_string in tqdm(embeddings_file): #this is a generator that streams lines one by one,
            #so each embedding_string is a line in embeddings_file
            # if (counter>check_point):
            #     print(str(int((float(counter)/float(file_size))*100))+r"%"+" of vocabulary vector dictionary completed")
            #     check_point+=slice
            # counter+=1
            embedding_vector = embedding_string.split()
            # print(embedding_string)
            # sys.exit()
            for word in vocab:
                # the first entry of embedding vector is the word
                if (word == embedding_vector[0]):
                    embeddings_list.append(embedding_vector)
        vectors_dict = dict()

        #with codecs.open (glove_file_name, 'r', 'utf-8') as embeddings_file: #This is to stream (using generators) when the file is large
        #embeddings_file= codecs.open (glove_file_name, 'r', 'utf-8')

            #now building a dictionary for words as keys and their corresponding vectors as values
        if verbose: print("Finished selecting embeddings for the vocabulary")
        if verbose: print("Started normalizing the digits and creating the vector dictionary")
        for embedding in embeddings_list:
            key = embedding[0]
            value = [np.float32(x) for x in embedding[1:]]
            vectors_dict.update({key:value})
            #vectors_dict.update({key:np.array(value)}) I need them as list for now so I commented this out,
            #I'll convert them to np.arrays later
        if verbose: print("Finished normalizing the digits and created the vector dictionary")
        return vectors_dict
        #works fine so far

    def build_data_case(self, i, tokens, history_range, future_range, corpus_size):
        #Helper for tokens_to_batches
        #build_training_case applies padding if required in the form of "PADDING_REQUIRED" for each added element to the list of
                #tokens that constitute the data for an occured a/the in the corpus. If padding is not
                #required, it simply builds the training case (history concatonated with the future tokens) and returns it.
        #i: token index
        #tokens: the list of all the tokens
        #history_range: range parameter for how far (in terms of # of tokens) to look back from the token
        #future_range: range parameter for how far (in terms of # of tokens) to look ahead from the token
        #corpus_size: the total number of tokens in the corpus
        data_case_string_list=[]
        if i<history_range:
            left_space=i
            history_tokens = self.get_history_tokens(i=i, left_space=left_space, tokens=tokens, padding=True)
            num_pads_required=history_range-left_space
            history_tokens = self.apply_left_padding(num_pads_required, history_tokens)
        else: #compute history tokens normally
            history_tokens = self.get_history_tokens(i=i, left_space=history_range, tokens=tokens, padding=False)
        if (i>(corpus_size-1-future_range)):
            #-1 is added because the indexing starts from 0 so if the corpus size is 300, for instance, the index of the 300th
            #element wil be 299.
            right_space=corpus_size-1-i
            future_tokens = self.get_future_tokens(i=i, right_space=right_space, tokens=tokens, padding=True)
            num_pads_required=future_range-right_space
            future_tokens = self.apply_right_padding(num_pads_required, future_tokens)
        else: #compute future tokens normally
            future_tokens=self.get_future_tokens(i=i, right_space=future_range, tokens=tokens, padding=False)
        data_case_string_list=history_tokens+future_tokens
        return data_case_string_list
    def get_future_tokens(self, i=0, right_space=0, tokens=[], padding=False):
        #Helper
        future_tokens = []
        for ii in range(right_space):
            j=i+(ii+1) #j is the future token index #ii+1 because "in range" spans from 0 to ii-1
            future_tokens.append(tokens[j])
        return future_tokens
    def get_history_tokens(self, i=0, left_space=0, tokens=[], padding=False):
        #Helper
        #Note: letf space semantically is either the full history range (the else statement) unless the words on the left are less than
            #the history range, and in that case it is whatever number of words left to the left (the if statement)
        #the defults are non-usable
        history_tokens = []
        if padding:
            for ii in range(left_space):
                history_tokens.append(tokens[ii])
        else:
            for ii in range(left_space):
                #history_tokens.append(tokens[i-(ii+1)])#this gets me the reverse order of words for the history
                history_tokens.append(tokens[i-(left_space-ii)])
        return history_tokens
    def apply_right_padding(self, num_pads_required=0, future_tokens=[]):
        #Helper
        for ii in range(num_pads_required):
            future_tokens.append("PADDING_REQUIRED")
        return future_tokens
    def apply_left_padding(self, num_pads_required=0, history_tokens=[]):
        #Helper
        for ii in range(num_pads_required):
            history_tokens.insert(0,"PADDING_REQUIRED")
        return history_tokens

    def tokens_to_dataset(self, vector_dict=dict(), tokens= [], history_range=2, future_range=2, corpus_size = 0, glove_file_name='',embeddings_dim=embeddings_dim, verbose=False):
        #tokens_length is the size of the corpus (number of tokens)
        #history_range: history parameter indicating how far back to look at in creating the training sequence
        #tokens is the list of the tokens of the text
        #corpus_size is the size of the corpus
        #glove_file_name is the name of the glove file used for embeddings
        dataset = [] #the returning argument, comprises a list of datapoints (i.e. concatonated context words embeddings)
                        #and a list of target values (0 for "a" and 1 for "the")
        if verbose: print("Creating data cases...")
        i=0
        vocab = list(set(tokens))
        data_cases=[]
        data_classes=[]
        for token in tokens:
            if self.a_or_the(token):
                case=[]
                correct_class=[]
                case = self.build_data_case(i, tokens, history_range, future_range, corpus_size) #resulting in case:["I", ... , "shop", "was", "closed"]
                data_cases.append(copy(case)) #resulting in data_cases:[[I, went, to, ... , shop, was closed],
                                                                                #[I, went, to, ... , shop, was closed]]
                data_classes.append(token) #resulting in data_classes:[a, the, a, the, the, the ...]
            i+=1
        if verbose: print("Data cases created.")

        if verbose: print("Converting to glove vectors")
        datapoints=[]
        dict_keys=vector_dict.keys()
        for case in data_cases:
            vec=[]
            for word in case:
                if (word=="PADDING_REQUIRED"):
                    vec+=(embeddings_dim*[np.float32(0)]) #zeros for paddings (a vec of zeros for the padded areas (words) in the case)
                elif(word in dict_keys): #"in vocab" returned keyError for "coach" although it's in the vocab, why? (debug later),
                                                #how come it hasn't found its way to the dict?
                    vec+=(vector_dict[word])
                else:
                    vec+=(embeddings_dim*[np.float32(0.1)]) #0.1's for OOV vectors
            datapoints.append(np.array(vec, np.float32))
        if verbose: print("Datapoints created")
        target_classes=[] # the list of target values (correct classes) corresponding to its data points by sharing the same index
        for data_class in data_classes:
            if(data_class=="a"):
                target_classes.append([0]) #warning may need to be changed to [0] instead of 0
                #alternatively: target_classes.append(np.float32(0))
            else:
                target_classes.append([1]) #warning may need to be changed to [1] instead of 1
                #alternatively: target_classes.append(np.float32(1))
        if verbose: print("Target classes created")

        dataset=[np.array(datapoints, np.float32), np.array(target_classes, np.int32)] #warning, removed .reshape(-1, 1) because the array is already 2 dimensional now
        return dataset
    def get_positions_iter(self, pattern=r'', text=''):
        #returns an iterator containing the position tuples of the matches of pattern in text
        p = re.compile(pattern)
        iter = p.finditer(text)
        return iter
    def positions_iter_to_spans(self, positions_iter):
        #turns position iterator into a list of spans
        l = []
        for item in positions_iter:
            l.append(item.span())
        return l
    def disambiguate(self, text='', span_list=[], model=None, vocab_vector_dict=None, word_range=2, verbose=False, pattern=r''):
        #text_size = len(text)
        capitalized=""
        replacement = '' #initialization
        revised_text=text #text initialization
        start=None
        end=None
        s_list=span_list #span_list initialization
        span_list_size=len(span_list)
        for span_i in range(len(span_list)):
            if (span_i==0):
                #inital span
                start=0
                end=s_list[span_i+1][0] #warnign this includes the first char of the next span but if used in a range will be ignored
            elif(span_i==(len(s_list)-1)):
                #final span
                start=s_list[span_i-1][1]
                end = len(revised_text)-1
            else:
                #other spans (in between initial and final)
                start = s_list[span_i-1][1]
                end = s_list[span_i+1][0]

            initial_letter=revised_text[(s_list[span_i])[0]+1]
            if (initial_letter.isupper()):
                capitalized="y"
            else:
                capitalized="n"
            tokens=nltk.tokenize.word_tokenize(revised_text[start:end].lower().decode("utf-8"))
            for token in tokens:
                if self.a_or_the(token):
                    datapoint = []
                    i = tokens.index(token)
                    datacase=[t for t in tokens if (tokens.index(t) in range(i-word_range, i+word_range+1) and tokens.index(t) != i)]
                    if len(datacase)==4:    #parameter warning word_range*2 could also be used for added autonomy
                        keys = vocab_vector_dict.keys()
                        for w in datacase:
                            if (w in keys):
                                datapoint+=vocab_vector_dict[w]
                            else:
                                datapoint+=50*[.1] #0.1's for OOV vectors
                        datapoint = np.array(datapoint, np.float32)
                        pred=model.predict(datapoint)
                        if (pred==np.array([1])): #warning may need to be changed back to 1
                            if (capitalized=="y"):
                                replacement=" The "
                            elif (capitalized=="n"):
                                replacement=" the "
                        elif(pred==np.array([0])):
                            if (capitalized=="y"):
                                replacement=" A "
                            elif (capitalized=="n"):
                                replacement=" a "
                        revised_text = revised_text[:(s_list[span_i])[0]] + replacement + revised_text[(s_list[span_i])[1]:]
                        positions_iter = self.get_positions_iter(pattern, revised_text)
                        s_list = self.positions_iter_to_spans(positions_iter) #updated the span list (i.e., s_list here) to reflect the changes
                                                                                        #in positions occured by implementing the fix in the current step
                        if verbose:
                            print("{} out of {} instances update so far".format(str(span_i), str(span_list_size)))
        return revised_text
    def pickle_file(self,object_to_picke,file_name,verbose=True):
        if verbose:
            print("Started creating vocab vec dictionary on {}".format(strftime("%a, %d %b %Y %H:%M:%S", localtime())))
        #pickling vocab_vec_dict_user
        if verbose: print("Saving file to {}".format(file_name))
        the_file = open(file_name, "wb")
        if verbose:
            print("Writing file to disk")
        pickle.dump(object_to_picke, the_file)
        if verbose:
            print("File saved successfully")
        the_file.close()
        return
