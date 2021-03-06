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

    # output a readable triples file
    # with open("train.txt", "w") as f:
    #     for triple_id in range(triples.shape[0]):
    #         h_id, r_id, t_id = triples[triple_id]
    #         h = i2e[h_id]
    #         r = i2r[r_id]
    #         t = i2e[t_id]
    #         f.write(h + "\t" + r + "\t" + t + "\n")

    # reduce triples to make reduced MRF
    # r_triples = np.ndarray(shape=(0, 3), dtype=int)
    # r_e2i = {}
    # r_i2e = {}
    # r_r2i = {}
    # r_i2r = {}
    # for triple_idx in range(triples.shape[0]):
    #     triple = triples[triple_idx]
    #     h = i2e[triple[0]]
    #     r = i2r[triple[1]]
    #     t = i2e[triple[2]]
    #     if "carrot" in h or "carrot" in t or "lime" in h or "lime" in t:
    #         r_triples = np.append(r_triples, [triple], axis=0)
    #         if triple[0] not in r_i2e:
    #             r_i2e[triple[0]] = i2e[triple[0]]
    #             r_e2i[i2e[triple[0]]] = triple[0]
    #         if triple[2] not in r_i2e:
    #             r_i2e[triple[2]] = i2e[triple[2]]
    #             r_e2i[i2e[triple[2]]] = triple[2]
    #         if triple[1] not in r_i2r:
    #             r_i2r[triple[1]] = i2r[triple[1]]
    #             r_r2i[i2r[triple[1]]] = triple[1]

    # generates the markov logic network
    mln = MLN(logic="FirstOrderLogic", grammar="PRACGrammar")
    # declares the constants
    for ent in e2i.keys():
        ent_type = ent.split("-")[-1]
        if ent_type == "r":
            mln.update_domain({"rooms": [ent]})
        elif ent_type == "l":
            mln.update_domain({"locations": [ent]})
        elif ent_type == "o":
            mln.update_domain({"objects": [ent]})
        elif ent_type == "a":
            mln.update_domain({"actions": [ent]})
        elif ent_type == "s":
            mln.update_domain({"states": [ent]})
        else:
            print("Error: Unknown entity type for domains!")
            exit()
    # declares the predicates
    rel2mln = {
        "HasEffect": [["actions", "states"], ["IsAction(?x)", "IsState(?y)"]],
        "InverseActionOf": [["actions", "actions"], ["IsAction(?x)", "IsAction(?y)"]],
        "InverseStateOf": [["states", "states"], ["IsState(?x)", "IsState(?y)"]],
        "ObjInRoom": [["objects", "rooms"], ["IsObject(?x)", "IsRoom(?y)"]],
        "LocInRoom": [["locations", "rooms"], ["IsLocation(?x)", "IsRoom(?y)"]],
        "ObjOnLoc": [["objects", "locations"], ["IsObject(?x)", "IsLocation(?y)"]],
        "ObjInLoc": [["objects", "locations"], ["IsObject(?x)", "IsLocation(?y)"]],
        "ObjCanBe": [["objects", "actions"], ["IsObject(?x)", "IsAction(?y)"]],
        "ObjUsedTo": [["objects", "actions"], ["IsObject(?x)", "IsAction(?y)"]],
        "ObjhasState": [["objects", "states"], ["IsObject(?x)", "IsState(?y)"]],
        "OperatesOn": [["objects", "objects"], ["IsObject(?x)", "IsObject(?y)"]]
    }
    for rel in r2i.keys():
        mln.predicate(Predicate(rel, rel2mln[rel][0]))
    mln.predicate(Predicate("IsRoom", ["rooms"]))
    mln.predicate(Predicate("IsLocation", ["locations"]))
    mln.predicate(Predicate("IsObject", ["objects"]))
    mln.predicate(Predicate("IsAction", ["actions"]))
    mln.predicate(Predicate("IsState", ["states"]))
    # declares the markov logic formulas in the markov logic network
    for pred in mln.iterpreds():
        if "Is" not in pred.name:
            mln << "0.0 " + rel2mln[pred.name][1][0] + " ^ " + rel2mln[pred.name][1][1] + " ^ " + pred.name + "(?x, ?y)"
    # loads the 'evidence' to learn markov logic network weights
    db = Database(mln)
    for ent in e2i.keys():
        ent_type = ent.split("-")[-1]
        if ent_type == "r":
            db << "IsRoom(" + ent + ")"
        elif ent_type == "l":
            db << "IsLocation(" + ent + ")"
        elif ent_type == "o":
            db << "IsObject(" + ent + ")"
        elif ent_type == "a":
            db << "IsAction(" + ent + ")"
        elif ent_type == "s":
            db << "IsState(" + ent + ")"
        else:
            print("Error: Unknown entity type for evidence!")
            exit()
    for triple_idx in range(triples.shape[0]):
        triple = triples[triple_idx]
        h = i2e[triple[0]]
        r = i2r[triple[1]]
        t = i2e[triple[2]]
        db << r + "(" + h + ", " + t + ")"

    # runs the learning on the markov logic network to get weights
    start_time = time.time()
    # learned_result = MLNLearn(mln=mln, db=db, verbose=True, save=True, method="BPLL_CG", output_filename="r_learned_weights.mln", multicore=True).run()
    # learned_result.tofile("r_learned_weights.mln")
    learned_result = MLN.load("learned_weights.mln")
    MLNQuery(queries="OperatesOn(sink-l,carrot-o)", verbose=True, mln=learned_result, db=db).run()
    print(" ---- %s seconds ---- " % (time.time() - start_time))
    pdb.set_trace()