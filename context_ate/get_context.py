import Preprocess
import os
from helpers import (
    terms2list
)
import pickle
import sys
import pandas as pd
from math import floor, ceil
import random
random.seed(10)
n_limit = 100
import numpy as np 
import itertools
flatten = itertools.chain.from_iterable

def add_sentences_to_dataset(word, text, dataset, label, window_size):
     # 5, 10, 20
    word_indices = [i for i, x in enumerate(text) if x == word]
    # n_limit = 10
    if limit:
        # to make sure it's not too imbalanced
        if len(word_indices) > n_limit:
            word_indices = random.sample(word_indices, n_limit)

    if len(word_indices) != 0:
        for word_index in word_indices:
            context_before = text[word_index-window_size:word_index]
            context_after = text[word_index+1:word_index+window_size+1]

            # use all after or all before if not enough
            if len(context_before) < window_size:
                # print("not enough before")
                n_tokens = window_size-len(context_before)
                # token = "<start>"
                token = "[UNK]"
                for i in range(n_tokens):
                    context_before.insert(0, token)
                # print(context_before)
                # input()

            elif len(context_after) < window_size:
                # print("not enough after")
                n_tokens = window_size-len(context_after)
                # token = "<end>"
                token = "[UNK]"
                for i in range(n_tokens):
                    context_after.append(token)
                # print(context_after)
                # input()

            assert len(context_before) == window_size and len(context_after) == window_size
            new_entry = [word, context_before, context_after, label]
            # print(new_entry)
            dataset.append(new_entry)
    return dataset

def add_context_to_dict(word_list, corpus_text_words, train_data, eval_data, label, window_size):
    train_pct = 0.7

    print(f"length: {len(word_list[:floor(len(word_list)*train_pct)])}, {len(word_list[ceil(len(word_list)*train_pct):])} ")
    # sys.exit()
    # print(len(np.array(corpus_text_words)[0]))
    corpus_array = list(flatten(corpus_text_words))
    print(len(corpus_array))
    # input()
    for word in word_list[:floor(len(word_list)*train_pct)]:
        counter = 0
        add_sentences_to_dataset(word, corpus_array, train_data, label, window_size)
    
    for word in word_list[ceil(len(word_list)*train_pct):]:
        counter = 0
        add_sentences_to_dataset(word, corpus_array, eval_data, label, window_size)
        # input()
    # sys.exit()
    print(len(train_data), len(eval_data))
    return train_data, eval_data

def get_context(correct_term_file, non_term_file, corpus_path, window_size):
    with open(corpus_separated_path, "rb") as f:
        corpus_text_words = pickle.load(f)

    correct_term_list = terms2list(correct_term_file)
    non_term_list = terms2list(non_term_file)

    ate_context_dataset = []
    train_data = []
    eval_data = []

    random.shuffle(correct_term_list)
    random.shuffle(non_term_list)

    train_data, eval_data = add_context_to_dict(correct_term_list, corpus_text_words, train_data, eval_data, 1, window_size)
    train_data, eval_data = add_context_to_dict(non_term_list, corpus_text_words, train_data, eval_data, 0, window_size)
    # print(ate_context_dataset)
    return train_data, eval_data

def list_to_csv(dataset, path):
    label_list = []
    text_list = []

    target_word_list = [] 
    context_before_list = []
    context_after_list = []

    for entry in dataset:
        target_word = entry[0]
        context_before = " ".join([item for item in entry[1]])
        context_after = " ".join([item for item in entry[2]])
        # combined = target_word + " <bfr> " + context_before + " <aft> " + context_after
        # combined = target_word + " [SEP] " + context_before + " [SEP] " + context_after
        combined =  context_before + " CT_BEGIN " + target_word + " CT_END " + context_after
        # print(combined)
        text_list.append(combined)
        label_list.append(entry[3])
        target_word_list.append(target_word)
        context_before_list.append(context_before)
        context_after_list.append(context_after)

    data = {
            "text": text_list,
            "labels": label_list,
            "target": target_word_list,
            "before": context_before_list,
            "after": context_after_list
            }

    df = pd.DataFrame(data=data)
    print(df.head())
    # df.to_excel("cyberate_dataset_extended.xlsx")
    df.to_csv(path)

if __name__ == "__main__":
    correct_term_file = os.path.abspath("data/terms/common_CorrectTerms.txt")
    non_term_file = os.path.abspath("data/terms/common_nonTerms.txt")
    corpus_separated_path = os.path.abspath("data/malware_texts_clean.p")
    limit = False

    window_size = 15

    train_data, eval_data = get_context(correct_term_file, non_term_file, corpus_separated_path, window_size)
    print(len(train_data))
    print(f"Ratio: {len(train_data)/(len(train_data)+ len(eval_data))}")
    

    if limit:
        path = f"data/new_data2/ate2_limit_{n_limit}"
    else:
        path = f"data/new_data2/ate2_{window_size}"

    if not os.path.exists(path):
        os.makedirs(path)

    train_path = os.path.join(path, "train.csv")
    eval_path = os.path.join(path, "eval.csv")
    list_to_csv(train_data, train_path)
    list_to_csv(eval_data, eval_path)
