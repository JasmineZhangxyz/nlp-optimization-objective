import numpy as np
import we
import json
from numpy.linalg import norm


def turn_word_embeddings_into_dic(embeddings_file):
    f = open(embeddings_file, 'r')
    lines = f.readlines()

    emb_dic = {}
    for line in lines:
        emb = line.split()
        emb_dic[emb[0]] = emb[1:]

    return emb_dic


def gender_direction(definitional, E):
    return we.doPCA(definitional, E).components_[0]


def cos_sim(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.dot(a, b) / (norm(a) * norm(b))


def calculate_direct_metric(N, g, emb_dic):
    '''
    This function calculates the direct bias, as defined in Bolukbasi et al. (2016).
    We set c=1 (hyperparameter)
    N = gender-neutral wordlist
    g = gender direction
    '''
    cosine_sum = 0
    for word in N:
        word_emb = emb_dic.get(word)
        cosine_sum += np.abs(cos_sim(word_emb, g))
    return cosine_sum / len(N)


if __name__ == "__main__":
    definitional_filename = "definitional_pairs.json"
    gender_neutral_filename = "gender_neutral_words.json"

    with open(definitional_filename, "r") as f:
        defs = json.load(f)

    with open(gender_neutral_filename, "r") as f:
        neutral = json.load(f)

    # ----- UNALTERED WORD2VEC MEASUREMENTS -----
    embeddings = "word2vec_no_reg.txt"
    E = we.WordEmbedding(embeddings)
    embedding_dic = turn_word_embeddings_into_dic(embeddings)
    gen_dir = gender_direction(defs, E)
    direct_bias = calculate_direct_metric(neutral, gen_dir, embedding_dic)

    # ----- OUR METHODS MEASUREMENTS -----
    o_embeddings = 'word2vec_our.txt'
    o_E = we.WordEmbedding(o_embeddings)
    o_embedding_dic = turn_word_embeddings_into_dic(o_embeddings)
    o_gender_direction = gender_direction(defs, o_E)
    o_direct_bias = calculate_direct_metric(neutral, o_gender_direction, o_embedding_dic)

    # ----- BOLUKBASI 2016 MEASUREMENTS -----
    b_embeddings = 'bolukbasi2016_debiased_word2vec.txt'
    b_E = we.WordEmbedding(b_embeddings)
    b_embedding_dic = turn_word_embeddings_into_dic(b_embeddings)
    b_gender_direction = gender_direction(defs, b_E)
    b_direct_bias = calculate_direct_metric(neutral, b_gender_direction, b_embedding_dic)

    # ----- GN GLOVE MEASUREMENTS -----
    g_embeddings = 'gnglove.txt'
    g_E = we.WordEmbedding(g_embeddings)
    g_embedding_dic = turn_word_embeddings_into_dic(g_embeddings)
    g_gender_direction = gender_direction(defs, g_E)
    g_direct_bias = calculate_direct_metric(neutral, g_gender_direction, g_embedding_dic)

    # ----- PRINT RESULTS -----
    print("\n----- DIRECT BIAS MEASUREMENT RESULTS -----")
    print("Unaltered word2vec Word Embeddings: " + str(round(direct_bias, 4)))
    print("Our Method Word Embeddings: " + str(round(o_direct_bias, 4)))
    print("Bolukbasi 2016 Word Embeddings: " + str(round(b_direct_bias, 4)))
    print("GN GLoVE Word Embeddings: " + str(round(g_direct_bias, 4)))
