from pymongo import MongoClient
from transformers import BertTokenizer, BertConfig, BertModel
import pickle
import os
import pandas as pd
import Utility, Preprocess
import numpy as np
import sys
from tqdm import tqdm
from gensim import corpora, models, similarities
from helpers import train_and_test_context, train_and_test_benchmark, train_and_test_siamese, train_and_test_concat_tads, train_and_test_siamese_local_global
# from helpers import train_and_test_new
import torch

# TODO method 2: https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/



def encode(tokenizer, model, text):
    input_ids = torch.tensor(tokenizer(text=text)["input_ids"]).unsqueeze(0) 
    output = model(input_ids)
    embedding = output.pooler_output.detach().numpy()
    embedding_array = np.squeeze(np.array(embedding, np.float32))
    return embedding_array 

def create_embedding_dataset(df, data_dir, out_path,  global_vec_generator, global_dict, global_dim,
                               local_vec_generator, local_model, local_dim, local_tokenizer, window_size):
    datapoints = []
    data_classes = []
    bert_list = ["bert", "bert_raw", "bert_finetune", "bert_refined"]

    if global_vec_generator == "bert":
        global_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        global_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for row in tqdm(df.itertuples()):
        row_split = row.text.split()
        context_before_list = row.before.split()
        context_before = " ".join(context_before_list[-window_size:])
        context_after_list = row.after.split()
        context_after = " ".join(context_after_list[:window_size])
        
        target_word = str(row.target)

        context = context_before + " CT_BEGIN " + target_word + " CT_END " + context_after 

        label = row.labels
        phrase_embedding = []

        if local_vec_generator in bert_list:
            try:
                input_ids = torch.tensor(local_tokenizer.encode(target_word)).unsqueeze(0)
                hidden_states = local_model(input_ids, output_hidden_states=True).hidden_states
                # use 2nd to last hidden state according to Han Xiao
                # to keep consistent with last paper, use last hidden state
                token_embeddings = hidden_states[-1].permute(1,0,2) # put tokens dimension in front

                tokens =[]

                for token_index, token in enumerate(token_embeddings):
                    # ignore CLS and SEP tokens at beginning or end 
                    if token_index not in [0, len(token_embeddings)-1]:
                        tokens.append(token)

                word_vector = sum(tokens)
                word_vector = [float(i) for i in list(word_vector.detach().numpy().flatten())]

                local_word_embedding_last = np.array(word_vector)

                token_embeddings = hidden_states[-2].permute(1,0,2) # put tokens dimension in front

                tokens =[]

                for token_index, token in enumerate(token_embeddings):
                    # ignore CLS and SEP tokens at beginning or end 
                    if token_index not in [0, len(token_embeddings)-1]:
                        tokens.append(token)

                word_vector = sum(tokens)
                word_vector = [float(i) for i in list(word_vector.detach().numpy().flatten())]

                local_word_embedding_second_last = np.array(word_vector)
                
                # local_word_embedding = encode(local_tokenizer, local_model, target_word)
                local_context_embedding = encode(local_tokenizer, local_model, context)
                
                # save separately, concat later!

                if global_vec_generator == "bert":
                    global_word_embedding = np.array(util.get_bert_global_vector_from_model(target_word, global_dict), np.float32)
                    global_context_embedding = encode(global_tokenizer, global_model, context)
    
                else:
                    global_context_embedding = []
                    for word in list(row_split):
                        global_vec = util.get_glove_global_vector_from_model(word, global_dict, global_dim)
                        global_context_embedding.append(global_vec)
                    global_context_embedding = np.mean(global_context_embedding)
                    global_context_embedding = np.array(global_context_embedding, np.float32)
                    
                    global_word_embedding = util.get_glove_global_vector_from_model(word, global_dict, global_dim)

            except:
                print("Failed")
                continue

            datapoint ={
                "local_word_lastlayer": local_word_embedding_last,
                "local_word_2ndlastlayer": local_word_embedding_second_last,
                "local_context": local_context_embedding,
                "global_word": global_word_embedding, 
                "global_context": global_context_embedding,
            }

        else:
            
            for context_word in context.split():
                word_embedding = util.new_word2localGlobalEmbeddings_models(context_word, global_vec_generator, global_dict, global_dim,
                                              local_vec_generator, local_model, local_dim, local_tokenizer=local_tokenizer)
                phrase_embedding.append(word_embedding)
            combined_embedding = np.array(phrase_embedding, np.float32)
            datapoint = {
                "combined": combined_embedding
            }
            # print(phrase_embedding.shape)
            # sys.exit()
        

        # local_dim + global_dim

        # print((phrase_len, local_dim+global_dim))
        # if phrase_array.shape != (phrase_len, local_dim+global_dim):
        #     input()

        # print(phrase_array.shape)
        datapoints.append(datapoint)
        data_classes.append(label)

    datapoints = np.array(datapoints)
    data_classes = np.array(data_classes)

    data_set= {
        "features": datapoints, 
        "labels": data_classes
    }

    # data_set_reshaped = [datapoints, data_classes.reshape(len(data_classes),)]

    preprocessor.pickle_file(data_set, os.path.join(data_dir, out_path))

    return data_set

# ["bert_raw", "bert_finetune"] w2v

# "bert_raw",
# local_vec_generators = ["bert_finetune", "bert_raw"]
local_vec_generators = ["bert_finetune"]
global_vec_generators = ["bert",]

client = MongoClient('localhost', 27017)
db = client['cyber_ate']
collection = db['ate2_oct']

train_path = "data/train.csv"
eval_path = "data/eval.csv"

local_only = False
contextual = True

all_data_file = "data/malware_sentences_raw.txt"
glove_dict_file_name="global_embedding_dicts/glove_global_dict.pkl"
bert_global_dict_file_name = "global_embedding_dicts/bert_global_dict.pkl"

util = Utility.Utility()
preprocessor = Preprocess.Preprocess()

window_sizes = [5, 10, 15]

for global_vec_generator in global_vec_generators:
    for local_vec_generator in local_vec_generators:
        local_bert_model_location_dict = {
            "bert_raw": "./bert_models/cyber_bert_model_raw",
            "bert_finetune": "./bert_models/cyber_bert_model_finetune_raw",
            "bert_refined": "./bert_models/cyber_bert_model_raw_new",
            "bert": "./bert_models/cyber_bert_model_raw",
        }

        try:
            local_bert_model_location = local_bert_model_location_dict[local_vec_generator]
        except:
            local_bert_model_location = None

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

        global_dict = pickle.load( open( global_vec_files_dict[global_vec_generator], "rb" ) )

        

        if global_vec_generator =="glove":
            global_dim = 50
        else:
            global_dim = 768

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

        for exp in range(5):
            for window_size in window_sizes:
                data_dir = os.path.abspath(f"embeddings/data_full/contextual_{global_vec_generator}_global_{local_vec_generator}_local_window{window_size}/")

                if (not os.path.exists(data_dir)):
                    os.mkdir(data_dir)
                    train_df = pd.read_csv(train_path)
                    eval_df = pd.read_csv(eval_path)

                    train_set = create_embedding_dataset(train_df, data_dir, "train.pkl",  global_vec_generator, global_dict, global_dim,
                                                local_vec_generator, local_model, local_dim, local_tokenizer=local_tokenizer, window_size=window_size)
                    eval_set = create_embedding_dataset(eval_df, data_dir, "eval.pkl",  global_vec_generator, global_dict, global_dim,
                                                local_vec_generator, local_model, local_dim, local_tokenizer=local_tokenizer, window_size= window_size)

                else:
                    if (not any(os.scandir(data_dir))):
                        train_df = pd.read_csv(train_path)
                        eval_df = pd.read_csv(eval_path)

                        train_set = create_embedding_dataset(train_df, data_dir, "train.pkl",  global_vec_generator, global_dict, global_dim,
                                                    local_vec_generator, local_model, local_dim, local_tokenizer=local_tokenizer, window_size=window_size)
                        eval_set = create_embedding_dataset(eval_df, data_dir, "eval.pkl",  global_vec_generator, global_dict, global_dim,
                                                    local_vec_generator, local_model, local_dim, local_tokenizer=local_tokenizer, window_size=window_size)
                # # "mlp", "svm",
                for classifier in ["mlp"]:
                    train_and_test_benchmark(classifier, local_vec_generator, global_vec_generator, collection, data_dir)
                    train_and_test_context(classifier, local_vec_generator, global_vec_generator, collection, data_dir, window_size)
                    train_and_test_concat_tads(classifier, local_vec_generator, global_vec_generator, collection, data_dir, window_size)
                classifier = "siamese"
                # train_and_test_siamese(classifier, local_vec_generator, global_vec_generator, collection, data_dir, window_size)
                train_and_test_siamese(classifier, local_vec_generator, global_vec_generator, collection, data_dir, window_size)

