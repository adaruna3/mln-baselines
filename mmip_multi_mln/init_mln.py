from itertools import combinations
from copy import copy
from argparse import ArgumentParser
import numpy as np

from pracmln import MLN
from pracmln.mln import Predicate
import data_utils as utils

import pdb


def get_domains(roles, instances):
    role_max_value_size = {role: [] for role in roles.keys()}
    for role in roles.keys():
        for instance in instances:
            inst_vals = []
            for inst_role, inst_value in instance:
                if inst_role == role:
                    inst_vals.append(inst_value.replace(" ","_"))
            if len(inst_vals) not in role_max_value_size[role]:
                role_max_value_size[role].append(len(inst_vals))
    domains = {role+"_d": set() for role in roles.keys()}
    for role, role_sizes in role_max_value_size.items():
        for role_size in role_sizes:
            if role_size == 0:
                domains[role+"_d"].add(tuple(["None"]))
                continue
            for value in combinations(roles[role], role_size):
                domains[role+"_d"].add(tuple(sorted(value)))
    return domains


def get_formulas(qrole, filename):
    auto = True if "auto" in filename else False
    data = utils.load_flattened_data(filename)
    corr_mat = np.asarray(data[1:])
    role2idx = {data[0][idx]: idx for idx in range(len(data[0]))}
    idx2role = {idx: data[0][idx] for idx in range(len(data[0]))}
    formulas = set()
    qidx = role2idx[qrole]
    corr_col = corr_mat[:,qidx]
    if auto:  # auto select formula for top 2 of matrix
        bidx = np.argpartition(corr_col, -1)[-1:]
    else:  # select all manually set formulas in matrix (non-zero)
        bidx = [idx for idx in range(corr_col.shape[0]) if corr_col[idx] > 0.0]
    for idx in bidx:
         formula = tuple(sorted([idx2role[idx]] + [qrole]))
         formulas.add(formula)
    return formulas


if __name__ == "__main__":
    parser = ArgumentParser(description="Initialize a new MLN")
    parser.add_argument("--input_datasets", type=str, help="(.txt)", nargs="*",
                        default=["./data/train_data.txt","./data/val_data.txt","./data/test_data.txt"])
    parser.add_argument("--roles_file", type=str, help="(.txt)", nargs="?",
                        default="./data/role_to_values.txt")
    parser.add_argument("--formula_file", type=str, help="(.mx)", nargs="?",
                        default="./data/formula_matrix_auto.mx")
    parser.add_argument("--output_mln", type=str, help="models", nargs="?",
                        default="./models/class.mln")
    parser.add_argument("--query_role", type=str, help="class,color,etc.", nargs="?",
                        default="class")
    args = parser.parse_args()
    # loads the data for MLN
    roles = utils.load_roles(args.roles_file)
    instances = []
    for dataset in args.input_datasets:
        instances += utils.load_flattened_data(dataset)
    role_constraints = utils.get_role_constraints(roles, instances)
    formulas = get_formulas(args.query_role, args.formula_file)
    domains = get_domains(roles, instances)
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
    mln.tofile(args.output_mln)
    print("The initial MLN has been written to '" + args.output_mln + "'.")
    print("Verify the MLN using the print out above.")
    print("NOTE: Weights of '" + args.output_mln + "' have NOT been learned.")
