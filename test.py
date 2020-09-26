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
        "HasEffect": [["actions", "states"], ["IsAction(+?x, Action)", "IsState(+?y, State)"]],
        "InverseActionOf": [["actions", "actions"], ["IsAction(+?x, Action)", "IsAction(+?y, Action)"]],
        "InverseStateOf": [["states", "states"], ["IsState(+?x, State)", "IsState(+?y, State)"]],
        "ObjInRoom": [["objects", "rooms"], ["IsObject(+?x, Object)", "IsRoom(+?y, Room)"]],
        "LocInRoom": [["locations", "rooms"], ["IsLocation(+?x, Location)", "IsRoom(+?y, Room)"]],
        "ObjOnLoc": [["objects", "locations"], ["IsObject(+?x, Object)", "IsLocation(+?y, Location)"]],
        "ObjInLoc": [["objects", "locations"], ["IsObject(+?x, Object)", "IsLocation(+?y, Location)"]],
        "ObjCanBe": [["objects", "actions"], ["IsObject(+?x, Object)", "IsAction(+?y, Action)"]],
        "ObjUsedTo": [["objects", "actions"], ["IsObject(+?x, Object)", "IsAction(+?y, Action)"]],
        "ObjhasState": [["objects", "states"], ["IsObject(+?x, Object)", "IsState(+?y, State)"]],
        "OperatesOn": [["objects", "objects"], ["IsObject(+?x, Object)", "IsObject(+?y, Object)"]]
    }
    for rel in r2i.keys():
        mln.predicate(Predicate(rel, rel2mln[rel][0]))
    mln.predicate(Predicate("IsRoom", ["rooms", "Room"]))
    mln.predicate(Predicate("IsLocation", ["locations", "Location"]))
    mln.predicate(Predicate("IsObject", ["objects", "Object"]))
    mln.predicate(Predicate("IsAction", ["actions", "Action"]))
    mln.predicate(Predicate("IsState", ["states", "State"]))
    # declares the markov logic formulas in the markov logic network
    for pred in mln.iterpreds():
        if "Is" not in pred.name:
            mln << "0.0 " + rel2mln[pred.name][1][0] + " ^ " + rel2mln[pred.name][1][1] + " ^ " + pred.name + "(+?x, +?y)"
    # loads the 'evidence' to learn markov logic network weights
    db = Database(mln)
    for ent in e2i.keys():
        ent_type = ent.split("-")[-1]
        if ent_type == "r":
            db << "IsRoom(" + ent + ", Room)"
        elif ent_type == "l":
            db << "IsLocation(" + ent + ", Location)"
        elif ent_type == "o":
            db << "IsObject(" + ent + ", Object)"
        elif ent_type == "a":
            db << "IsAction(" + ent + ", Action)"
        elif ent_type == "s":
            db << "IsState(" + ent + ", State)"
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
    result = MLNLearn(mln=mln, db=db, verbose=True, save=True, multicore=True, profile=True).run()
    print(" ---- %s seconds ---- " % (time.time() - start_time))
    #result = MLNQuery(verbose=True, mln=mln, db=db).run()
    pdb.set_trace()