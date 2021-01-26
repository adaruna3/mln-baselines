from copy import copy
import json
import numpy as np

from pracmln import MLN
from pracmln.mln import Predicate
from data_utils import load_flattened_data

import pdb


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


def get_domains(roles, instances):
    domains = {role+"_d": set() for role in roles.keys()}
    for role in roles.keys():
        for instance in instances:
            inst_vals = []
            for inst_role, inst_value in instance:
                if inst_role == role:
                    inst_vals.append(inst_value)
            domains[role+"_d"].add(tuple(sorted(inst_vals)))
    return domains


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
    domains = get_domains(roles, tr+va+te)
    # generates the markov logic network
    mln = MLN(logic="FirstOrderLogic", grammar="PRACGrammar")
    for domain, values in domains.items():  # domains
        for value in values:
            if len(value) > 1:
                const = ','.join(value)
            elif len(value) > 0:
                const = value[0]
            else:
                const = "None"
            mln.update_domain({domain: [const]})
    pdb.set_trace()
    for role in roles.keys():  # predicates
        pred = Predicate(role, [role+"_d!"])
        pdb.set_trace()
        mln.predicate(pred)  # hard-functional constraints only
    pdb.set_trace()
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
    pdb.set_trace()
    print("The initial MLN has been written to 'initial.mln'.")
    print("Verify the MLN using the print out above.")
    print("NOTE: Weights of 'initial.mln' have NOT been learned.")
