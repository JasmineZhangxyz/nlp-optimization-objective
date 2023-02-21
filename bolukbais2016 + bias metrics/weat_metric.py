import numpy as np
from numpy.linalg import norm


def turn_word_embeddings_into_dic(embeddings_file):
    f = open(embeddings_file, 'r')
    lines = f.readlines()

    emb_dic = {}
    for line in lines:
        emb = line.split()
        emb_dic[emb[0]] = emb[1:]
    return emb_dic


def cos_sim(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.dot(a, b) / (norm(a) * norm(b))


def s_word_A_B(emb_dic, word, A, B):
    a_sum = 0
    b_sum = 0
    word_emb = emb_dic.get(word)

    for a in A:
        a_emb = emb_dic.get(a)
        a_sum += cos_sim(word_emb, a_emb)

    for b in B:
        b_emb = emb_dic.get(b)
        b_sum +=cos_sim(word_emb, b_emb)

    return (a_sum / len(A)) - (b_sum / len(B))


def s_X_Y_A_B(emb_dic, X, Y, A, B):
    sum_X = 0
    sum_Y = 0

    for x in X:
        sum_X += s_word_A_B(emb_dic, x, A, B)

    for y in Y:
        sum_Y += s_word_A_B(emb_dic, y, A, B)

    return abs(sum_X - sum_Y)


if __name__ == "__main__":
    # "male" professions
    A = ["engineer", "entrepreneur", "inventor", "doctor", "manager", "mathematician"]
    # "female" professions
    B = ["receptionist", "teacher", "dancer", "nanny", "nurse", "homemaker"]
    # male + female lists
    X = ["man", "boy", "he", "father", "son", "guy", "male", "his", "himself"]
    Y = ["woman", "girl", "she", "mother", "daughter", "gal", "female", "her", "herself"]

    # ----- UNALTERED WORD2VEC MEASUREMENTS -----
    embeddings = "word2vec_no_reg.txt"
    embedding_dic = turn_word_embeddings_into_dic(embeddings)
    weat_metric = s_X_Y_A_B(embedding_dic, X, Y, A, B)

    # ----- OUR METHODS MEASUREMENTS -----
    o_embeddings = 'word2vec_our.txt'
    o_embedding_dic = turn_word_embeddings_into_dic(o_embeddings)
    o_weat_metric = s_X_Y_A_B(o_embedding_dic, X, Y, A, B)

    # ----- BOLUKBASI 2016 MEASUREMENTS -----
    b_embeddings = 'bolukbasi2016_debiased_word2vec.txt'
    b_embedding_dic = turn_word_embeddings_into_dic(b_embeddings)
    b_weat_metric = s_X_Y_A_B(b_embedding_dic, X, Y, A, B)

    # ----- GN GLOVE MEASUREMENTS -----
    g_embeddings = 'gnglove.txt'
    g_embedding_dic = turn_word_embeddings_into_dic(g_embeddings)
    g_weat_metric = s_X_Y_A_B(g_embedding_dic, X, Y, A, B)

    # ----- PRINT RESULTS -----
    print("\n----- WEAT METRIC MEASUREMENT RESULTS -----")
    print("Unaltered word2vec Word Embeddings: " + str(round(weat_metric, 4)))
    print("Our Method Word Embeddings: " + str(round(o_weat_metric, 4)))
    print("Bolukbasi 2016 Word Embeddings: " + str(round(b_weat_metric, 4)))
    print("GN GLoVE Word Embeddings: " + str(round(g_weat_metric, 4)))
