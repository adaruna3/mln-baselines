from pracmln import MLN, MLNQuery, MLNLearn
from pracmln.mln import Predicate, Database
import pandas as pd
import numpy as np
import time

import pdb


def load_id_map(label_file):
    try:
        labels = pd.read_csv(label_file, sep="\t", skiprows=1, header=None,
                             dtype={0: np.str, 1: np.int32})
    except IOError as e:
        print("Could not load " + str(label_file), "f")
        raise IOError

    label2index = {labels.iloc[idx, 0]: labels.iloc[idx, 1] for idx in range(len(labels))}
    index2label = {labels.iloc[idx, 1]: labels.iloc[idx, 0] for idx in range(len(labels))}
    return label2index, index2label


def load_triple_set(names, unique=False):
    if type(names) == str:
        names = [names]
    triples = load_triples([name + ".txt" for name in names])
    if unique:
        triples = np.unique(triples, axis=0)
    return triples


def load_triples(triples_files):
    triples = np.ndarray(shape=(0, 3), dtype=int)
    for triples_file in triples_files:
        try:
            file_triples = pd.read_csv(triples_file, sep=" |,", skiprows=1, header=None,
                                       dtype={0: np.int32, 1: np.int32, 2: np.int32}, engine="python").to_numpy()
            file_triples[:, [1, 2]] = file_triples[:, [2, 1]]
            triples = np.append(triples, file_triples, axis=0)
        except IOError as e:
            print('Could not load ' + str(triples_file), "f")
            raise IOError
    return triples

if __name__ == "__main__":
    # loads the data for MLN
    e2i, i2e = load_id_map("entity2id.txt")
    e2i = {label.replace(".", "-"): id for label, id in e2i.items()}
    i2e = {id: label.replace(".", "-") for id, label in i2e.items()}
    if len(e2i.keys()) != len(set(e2i.keys())):
        print("Error, some repeated rels.")
    r2i, i2r = load_id_map("relation2id.txt")
    if len(r2i.keys()) != len(set(r2i.keys())):
        print("Error, some repeated rels.")
    triples = load_triple_set("train2id", unique=True)

    # generates the markov logic network
    mln = MLN(logic="FuzzyLogic", grammar="PRACGrammar")
    # declares the constants
    for ent in e2i.keys():
        mln.update_domain({"ents": [ent]})
    # declares the predicates
    for rel in r2i.keys():
        mln.predicate(Predicate(rel, ["ents", "ents"]))
    mln.predicate(Predicate("IsA", ["ents", "entity"]))
    # declares the markov logic formulas in the markov logic network
    for pred in mln.iterpreds():
        if "IsA" not in pred.name:
            mln << "0.0 IsA(+?x, entity) ^ IsA(+?y, entity) ^ " + pred.name + "(+?x, +?y)"

    # loads the 'evidence' to learn markov logic network weights
    db = Database(mln)
    for ent in e2i.keys():
        db << "IsA(" + ent + ", entity)"
    for triple_idx in range(triples.shape[0]):
        triple = triples[triple_idx]
        h = i2e[triple[0]]
        r = i2r[triple[1]]
        t = i2e[triple[2]]
        db << r + "(" + h + ", " + t + ")"
    # runs the learning on the markov logic network to get weights
    start_time = time.time()
    result = MLNLearn(mln=mln, db=db, verbose=True, save=True, multicore=True, profile=True).run()
    print(" ---- %s seconds ---- " % (time.time() - start_time))
    #result = MLNQuery(verbose=True, mln=mln, db=db).run()
    pdb.set_trace()