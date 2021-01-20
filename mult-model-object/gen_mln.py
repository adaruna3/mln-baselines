from pracmln import MLN, MLNQuery, MLNLearn
from pracmln.mln import Predicate, Database
import pandas as pd
import numpy as np
import time
import json
from copy import copy

import pdb

def load_flattened_data(filename):
    flattened_data = []
    with open(filename, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                flattened_data.append(eval(line))
    return flattened_data

def get_role_constraints(roles, instances):
    hard_roles = {}
    soft_roles = {}
    role_count_init = {}

    for role in roles.keys():
        hard_roles[role] = 1
        soft_roles[role] = 1
        role_count_init[role] = 0

    for instance in instances:
        role_count = copy(role_count_init)
        for role, value in instance:
            role_count[role] += 1

        for role in roles:
            if role_count[role] != 1:
                hard_roles[role] = 0
                if role_count[role] > 1:
                    soft_roles[role] = 0

    role_constraints = {}
    for role in roles:
        if hard_roles[role]:
            role_constraints[role] = '!'
        elif soft_roles[role]:
            role_constraints[role] = '?'
        else:
            role_constraints[role] = ''

    return role_constraints


if __name__ == "__main__":
    # loads the data for MLN
    with open("role_to_values.json", "r") as f:
        roles = json.loads(f.readlines()[0])
    tr = load_flattened_data("train_data.txt")
    va = load_flattened_data("val_data.txt")
    te = load_flattened_data("test_data.txt")
    role_constraints = get_role_constraints(roles, tr + va + te)
    # generates the markov logic network
    mln = MLN(logic="FirstOrderLogic", grammar="PRACGrammar")
    for role, values in roles.items(): # constants
        for value in values:
            mln.update_domain({role+"_d": [value]})
    for role in roles.keys(): # predicates
        mln.predicate(Predicate(role, [role+"_d" + role_constraints[role]])) # hard-, soft-, and no-functional constraints
    pdb.set_trace()

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