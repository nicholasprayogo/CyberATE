import Preprocess
import os
from helpers import (
    terms2list
)
import pickle

def add_context_to_dict(word_list, corpus_text_words, context_dict):
    window_size = 5
    for word in word_list:
        for text in corpus_text_words:
            try: # raises error if not found, so move on to next text
                word_index = text.index(word)
                break

            except:
                continue
        context_before = text[word_index-window_size:word_index]
        context_after = text[word_index+1:word_index+window_size+1]

        # use all after or all before if not enough
        if len(context_before) < window_size:
            # print("not enough before")
            context_before = context_after
            context_after = text[word_index+window_size+1:word_index+window_size+1+window_size]

        elif len(context_after) < window_size:
            # print("not enough after")
            context_after = context_before
            context_before = text[word_index-window_size*2:word_index-window_size]

        assert len(context_before) == window_size and len(context_after) == window_size

        context_dict[word] = [context_before, context_after]
        # print(word)
        # print(context_before)
        # print(context_after)
        # input()

    return context_dict

def get_context(correct_term_file, non_term_file, corpus_path):
    with open(corpus_separated_path, "rb") as f:
        corpus_text_words = pickle.load(f)

    correct_term_list = terms2list(correct_term_file)
    non_term_list = terms2list(non_term_file)

    context_dict = {}

    context_dict = add_context_to_dict(correct_term_list, corpus_text_words, context_dict)
    context_dict = add_context_to_dict(non_term_list, corpus_text_words, context_dict)

if __name__ == "__main__":
    correct_term_file = os.path.abspath("data/terms/common_CorrectTerms.txt")
    non_term_file = os.path.abspath("data/terms/common_nonTerms.txt")
    corpus_separated_path = os.path.abspath("data/malware_texts_clean.p")
    context_dict = get_context(correct_term_file, non_term_file, corpus_separated_path)
    with open("data/context_dict.p", "wb") as f:
        pickle.dump(context_dict, f)
