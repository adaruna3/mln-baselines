import sys
from copy import deepcopy
from argparse import ArgumentParser
from time import time

from pracmln import MLN, MLNQuery
from pracmln.mln import Database
import data_utils as utils

import pdb


def extract_rv(db, predname):
    # gets the role-value pair(s) in database
    values = []
    for atom, _ in dict(db._evidence).items():
        _, pred, const = db.mln.logic.parse_literal(atom)
        if pred == predname: values.append(const[0])
    return values


def generate_test_dbs(role, dbs):
    test_dbs = deepcopy(dbs)
    idx = 0
    while idx < len(test_dbs):
        values = extract_rv(test_dbs[idx], role)
        if len(values) != 1:
            if len(values) == 0:
                del test_dbs[idx]
                continue
            else:
                pdb.set_trace()
                print("error: db case not handled.")
                exit()
        test_dbs[idx].retractall(role)
        idx += 1
    return test_dbs


def extract_predicted(mln, results):
    assert sum(results.values()) == 1, "wcsp results are empty"
    for atom, belief in results.items():
        if belief:
            return mln.logic.parse_literal(atom)[2][0]


def score_mln(mln, role, test_dbs):
    num_queries = len(mln.domains[role+"_d"])
    domain_values = set(deepcopy(mln.domains[role+"_d"]))
    ranks = {}
    for idx in range(len(test_dbs)):
        db = test_dbs[idx]
        instance_ranks = {}
        for idq in range(num_queries-1):
            try:
                wcsp = MLNQuery(queries=role, verbose=False, mln=mln, 
                                db=db, method="WCSPInference").run()
                predicted = extract_predicted(mln, wcsp.results)
            except AssertionError as e:
                print(e)
                pdb.set_trace()
            instance_ranks[predicted] = num_queries - idq
            db[role + "(" + predicted + ")"] = 0.0
        missing = domain_values.difference(set(instance_ranks.keys()))
        try:
            assert len(missing) == 1, "missing queries"
            instance_ranks[list(missing)[0]] = 1
        except AssertionError as e:
            print(e)
            pdb.set_trace()
        ranks[idx] = deepcopy(instance_ranks)
    return ranks


def scores2instance_scores(query_role, roles, positives, negatives, scores):
    examples = utils.mln_get_instance_examples_dict(positives, negatives, query_role, roles, True)
    instance_scores = {}
    for idx in range(len(examples)):
        for value in examples[idx].keys():
            try:
                instance_scores[examples[idx][value]['rv']] = scores[idx][value]
            except KeyError as e:
                # negative is missing, rank lowest TODO VERIFY!!
                instance_scores[examples[idx][value]['rv']] = 1
    return instance_scores


if __name__ == "__main__":
    parser = ArgumentParser(description="Learn MLN")
    parser.add_argument("--input_mln", type=str, help="(.mln)", nargs="?",
                        default="./models/learned.mln")
    parser.add_argument("--positive_database", type=str, help="(.db)", nargs="?",
                        default="./data/valid.db")
    parser.add_argument("--positive_dataset", type=str, help="(.txt)", nargs="?",
                        default="./data/valid_data.txt")
    parser.add_argument("--negative_dataset", type=str, help="(.txt)", nargs="?",
                        default="./data/val_data_negative.txt")
    parser.add_argument("--roles_file", type=str, help="(.txt)", nargs="?",
                        default="./data/role_to_values.txt")
    args = parser.parse_args()
    # loads the MLN, DBs, and instances
    with open(args.roles_file, "r") as f:
        roles = eval(f.readlines()[0])
    mln = MLN.load(args.input_mln)
    dbs = Database.load(mln, args.positive_database)
    p_examples = utils.load_flattened_data(args.positive_dataset)
    n_examples = utils.load_flattened_data(args.negative_dataset)
    test_times = []
    # begins testing roles
    for role in roles.keys():
        start = time()
        # creates testing DBs with labels
        test_dbs = generate_test_dbs(role, dbs)
        # gets MLN scores
        scores = score_mln(mln, role, test_dbs)
        # makes instance-score datastructure
        instance_scores = scores2instance_scores(role, roles, p_examples, n_examples, scores)
        # gets metrics for the role
        utils.compute_metric_scores(p_examples, n_examples, instance_scores, [role], roles, save_dir="./results")
        test_times.append(time()-start)
    for r_idx in range(len(roles.keys())):
        duration = int( test_times[r_idx] / 60.0)
        role = [r for r in roles.keys()][r_idx]
        print("Testing " + str(role) + " took " + str(duration) + " minutes.")
