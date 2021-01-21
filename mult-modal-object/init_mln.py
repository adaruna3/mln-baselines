from pracmln import MLN
from pracmln.mln import Predicate
import numpy as np
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


def get_formulas(filename):
    data = load_flattened_data(filename)
    corr_mat = np.asarray(data[1:])
    role2idx = {data[0][idx]: idx for idx in range(len(data[0]))}
    idx2role = {idx: data[0][idx] for idx in range(len(data[0]))}
    formulas = set()
    for qrole, qidx in role2idx.items():
        corr_col = corr_mat[:,qidx]
        bidx = np.argpartition(corr_col, -1)[-1:]
        formula = tuple(sorted([idx2role[idx] for idx in bidx] + [qrole]))
        formulas.add(formula)
    return formulas


if __name__ == "__main__":
    # loads the data for MLN
    with open("role_to_values.json", "r") as f:
        roles = json.loads(f.readlines()[0])
    tr = load_flattened_data("train_data.txt")
    va = load_flattened_data("val_data.txt")
    te = load_flattened_data("test_data.txt")
    role_constraints = get_role_constraints(roles, tr + va + te)
    formulas = get_formulas("correlation_matrix.txt")

    # generates the markov logic network
    mln = MLN(logic="FirstOrderLogic", grammar="PRACGrammar")
    for role, values in roles.items():  # constants
        for value in values:
            mln.update_domain({role+"_d": [value]})
    for role in roles.keys():  # predicates
        mln.predicate(Predicate(role, [role+"_d" + role_constraints[role]]))  # hard-, soft-, and no-functional constraints
    for formula in formulas:  # formulas
        formula_str = "0.0 "
        for idx in range(len(formula)):
            role = formula[idx]
            if idx+1 < len(formula):
                formula_str += role + "(+?" + role[:2] + ")" + " ^ "
            else:
                formula_str += role + "(+?" + role[:2] + ")"
        mln << formula_str
    mln.write()
    mln.tofile("initial.mln")
    print("The initial MLN has been written to 'initial.mln'.")
    print("Verify the MLN using the print out above.")
    print("NOTE: Weights of 'initial.mln' have NOT been learned.")
