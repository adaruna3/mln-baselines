from copy import copy
from argparse import ArgumentParser
import json

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
    test_dbs = copy(dbs)
    for db in test_dbs:
        values = extract_rv(db, role)
        if len(values) != 1:
            print("error: db case not handled.")
            exit()
        db.retractall(role)
    return test_dbs


def extract_predicted(mln, results):
    assert sum(results.values()) == 1
    for atom, belief in results.items():
        if belief:
            return mln.logic.parse_literal(atom)[2][0]


def score_mln(mln, role, test_dbs):
    num_queries = len(mln.domains[role+"_d"])
    ranks = {}
    for idx in range(len(test_dbs)):
        db = test_dbs[idx]
        instance_ranks = {}
        for idq in range(num_queries):
            wcsp = MLNQuery(queries="class", verbose=False, mln=mln, db=db, method="WCSPInference").run()
            predicted = extract_predicted(mln, wcsp.results)
            instance_ranks[predicted] = num_queries - idq
            db[role + "(" + predicted + ")"] = 0.0
        ranks[idx] = copy(instance_ranks)
    return ranks


def scores2instance_scores(query_role, roles, positives, negatives, scores):
    examples = utils.mln_get_instance_examples_dict(positives, negatives, query_role, roles, True)
    instance_scores = {}
    for idx in range(len(examples)):
        for value in examples[idx].keys():
            instance_scores[examples[idx][value]['rv']] = scores[idx][value]
    return instance_scores

if __name__ == "__main__":
    parser = ArgumentParser(description="Learn MLN")
    parser.add_argument('input_mln', type=str, help='(.mln)')
    parser.add_argument('positive_database', type=str, help='(.db)')
    parser.add_argument('positive_dataset', type=str, help='(.txt)')
    parser.add_argument('negative_dataset', type=str, help='(.txt)')
    args = parser.parse_args()

    # loads the MLN, DBs, and instances
    with open("role_to_values.json", "r") as f:
        roles = json.loads(f.readlines()[0])
    mln = MLN.load(args.input_mln)
    dbs = Database.load(mln, args.positive_database)
    p_examples = utils.load_flattened_data(args.positive_dataset)
    n_examples = utils.load_flattened_data(args.negative_dataset)
    # begins testing acceptable roles
    testable_roles = ["class"]
    for role in testable_roles:
        # creates testing DBs with labels
        test_dbs = generate_test_dbs(role, dbs)
        # gets MLN scores
        scores = score_mln(mln, role, test_dbs)
        # makes instance-score datastructure
        instance_scores = scores2instance_scores(role, roles, p_examples, n_examples, scores)
        # gets metrics for the role
        utils.compute_metric_scores(p_examples, n_examples, instance_scores, [role], roles, save_dir="./result")