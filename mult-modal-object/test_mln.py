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
    labels = []
    test_dbs = copy(dbs)
    for db in test_dbs:
        values = extract_rv(db, role)
        if len(values) != 1:
            print("error: db case not handled.")
            exit()
        labels.append(values[0])
        db.retractall(role)
    return labels, test_dbs


def extract_predicted(mln, results):
    assert sum(results.values()) == 1
    for atom, belief in results.items():
        if belief:
            return mln.logic.parse_literal(atom)[2][0]


def evaluate(mln, test_dbs, labels):
    hits1 = 0
    for idx in range(len(labels)):
        label = labels[idx]
        db = test_dbs[idx]
        wcsp = MLNQuery(queries="class", verbose=False, mln=mln, db=db, method="WCSPInference").run()
        predicted = extract_predicted(mln, wcsp.results)
        if predicted == label:
            hits1 += 1
    return float(hits1) / float(len(labels))


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
    testable_roles = ["class"]
    for role in testable_roles:
        # creates testing DBs with labels
        labels, test_dbs = generate_test_dbs(role, dbs)
        # evaluate
        hits1 = evaluate(mln, test_dbs, labels)
        print("Hits@1 is " + str(hits1))

    # p_examples, n_examples, roles
    # result = utils.perturb_positive_instance_for_evaluation(p_examples, n_examples, query_role, roles, True)