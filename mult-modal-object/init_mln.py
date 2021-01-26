from copy import copy
import json
import numpy as np

from pracmln import MLN
from pracmln.mln import Predicate
from data_utils import load_flattened_data, get_role_constraints

import pdb


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
                const = ''.join(value)
            elif len(value) > 0:
                const = value[0]
            else:
                const = "None"
            mln.update_domain({domain: [const]})
    for role in roles.keys():  # predicates
        mln.predicate(Predicate(role, [role + "_d!"]))  # hard-functional constraints only
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
