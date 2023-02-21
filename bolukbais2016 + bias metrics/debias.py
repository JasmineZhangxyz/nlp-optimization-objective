from __future__ import print_function, division
import we
import json
import numpy as np
import sys
if sys.version_info[0] < 3:
    import io
    open = io.open
"""
Hard-debias embedding
Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""


def debias(E, gender_specific_words, definitional, equalize):
    gender_direction = we.doPCA(definitional, E).components_[0]
    #f = open("bolukbasi_debiased_gender_direction.txt", "w")
    #f.write(str(gender_direction))
    #f.close()

    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    E.normalize()
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    print(candidates)
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = we.drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()


if __name__ == "__main__":

    embedding_filename = "word2vec_no_reg.txt"
    definitional_filename = "definitional_pairs.json"
    gendered_words_filename = "gendered_words.json"
    equalize_filename = "equalized_pairs.json"
    debiased_filename = "bolukbasi2016_debiased_word2vec.txt"

    with open(definitional_filename, "r") as f:
        defs = json.load(f)
    print("definitional", defs)

    with open(equalize_filename, "r") as f:
        equalize_pairs = json.load(f)

    with open(gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)
    print("gender specific", len(gender_specific_words), gender_specific_words[:10])

    E = we.WordEmbedding(embedding_filename)

    print("Debiasing...")
    debias(E, gender_specific_words, defs, equalize_pairs)

    print("Saving to file...")
    if embedding_filename[-4:] == debiased_filename[-4:] == ".bin":
        E.save_w2v(debiased_filename)
    else:
        E.save(debiased_filename)

    print("\n\nDone!\n")
