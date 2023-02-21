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


def indirect_bias(g, w, v):
    g = np.array(g, dtype=float)
    w = np.array(w, dtype=float)
    v = np.array(v, dtype=float)

    wg = np.dot(np.dot(w, g), g)
    vg = np.dot(np.dot(v, g), g)
    w_norm = norm(w - wg)
    v_norm = norm(v - vg)

    return (np.dot(w, v) - np.dot(w - wg, v - vg) / (w_norm * v_norm)) / np.dot(w, v)


def compare_words_indirect_metric(g, emb_dic, def_pair, professions):
    closer_to_female = []
    closer_to_male = []
    a_vec = emb_dic.get(def_pair[0])
    b_vec = emb_dic.get(def_pair[1])
    for word in professions:
        word_emb = emb_dic.get(word)
        dist_a = indirect_bias(g, a_vec, word_emb)
        dist_b = indirect_bias(g, b_vec, word_emb)
        if dist_a < dist_b:
            closer_to_female.append(word)
        else:
            closer_to_male.append(word)
    return closer_to_female, closer_to_male


def indirect_metric_results():
    not_f_anymore = []      # profession in f1 not found in f2 anymore
    both_f = []             # profession in f1 and f2
    not_m_anymore = []      # profession in m1 not found in m2 anymore
    both_m = []             # profession in m1 and m2

    for prof in f1:
        if prof not in f2:
            not_f_anymore.append(prof)
        else:
            both_f.append(prof)

    for prof in m1:
        if prof not in m2:
            not_m_anymore.append(prof)
        else:
            both_m.append(prof)

    return not_f_anymore, not_m_anymore


if __name__ == "__main__":
    definitional_filename = "definitional_pairs.json"
    gender_neutral_filename = "gender_neutral_words.json"

    with open(definitional_filename, "r") as f:
        defs = json.load(f)

    with open(gender_neutral_filename, "r") as f:
        neutral = json.load(f)

    # ------------- UNALTERED WORD2VEC MEASUREMENTS -------------
    # embeddings = "word2vec_no_reg.txt"
    # E = we.WordEmbedding(embeddings)
    # embedding_dic = turn_word_embeddings_into_dic(embeddings)
    # gen_dir = gender_direction(defs, E)

    # indirect_bias = compare_words_indirect_metric(gen_dir, embedding_dic, ["softball", "football"], neutral)
    # print("----- INDIRECT BIAS MEASUREMENT FOR UNALTERED WORD2VEC WORD EMBEDDINGS -----")
    # print("closer to softball (female): " + str(indirect_bias[0]))
    # print("closer to football (male): " + str(indirect_bias[1]))

    # --------------- OUR METHODS MEASUREMENTS ---------------
    # o_embeddings = 'word2vec_our.txt'
    # o_E = we.WordEmbedding(o_embeddings)
    # o_embedding_dic = turn_word_embeddings_into_dic(o_embeddings)
    # o_gender_direction = gender_direction(defs, o_E)
    #
    # o_indirect_bias = compare_words_indirect_metric(o_gender_direction, o_embedding_dic, ["softball", "football"], neutral)
    # print("----- INDIRECT BIAS MEASUREMENT FOR OUR METHOD WORD EMBEDDINGS -----")
    # print("closer to softball (female): " + str(o_indirect_bias[0]))
    # print("closer to football (male): " + str(o_indirect_bias[1]))

    # --------------- BOLUKBASI 2016 MEASUREMENTS ---------------
    # b_embeddings = 'bolukbasi2016_debiased_word2vec.txt'
    # b_E = we.WordEmbedding(b_embeddings)
    # b_embedding_dic = turn_word_embeddings_into_dic(b_embeddings)
    # b_gender_direction = gender_direction(defs, b_E)

    # b_indirect_bias = compare_words_indirect_metric(b_gender_direction, b_embedding_dic, ["softball", "football"], neutral)
    # print("----- INDIRECT BIAS MEASUREMENT FOR BOLUKBASI 2016 WORD EMBEDDINGS -----")
    # print("closer to softball (female): " + str(b_indirect_bias[0]))
    # print("closer to football (male): " + str(b_indirect_bias[1]))

    # --------------- GN GLOVE MEASUREMENTS ---------------
    g_embeddings = 'gnglove.txt'
    g_E = we.WordEmbedding(g_embeddings)
    g_embedding_dic = turn_word_embeddings_into_dic(g_embeddings)
    g_gender_direction = gender_direction(defs, g_E)

    g_indirect_bias = compare_words_indirect_metric(g_gender_direction, g_embedding_dic, ["softball", "football"], neutral)
    print("----- INDIRECT BIAS MEASUREMENT FOR GN GLOVE WORD EMBEDDINGS -----")
    print("closer to softball (female): " + str(g_indirect_bias[0]))
    print("closer to football (male): " + str(g_indirect_bias[1]))

