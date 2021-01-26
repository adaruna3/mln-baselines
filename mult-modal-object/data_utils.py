import os
import numpy as np
import copy
import pandas as pd
from scipy.stats import rankdata

import pdb


def compute_metric_scores(positive_data, negative_data, instance_prediction, query_roles, role_to_values,
                          ignore_non_object=True, save_dir=None, verbose=True):
    """
    This function computes metric scores

    :param positive_data: a list of positive instances
    :param negative_data: a list of negative instances
    :param instance_prediction: a dictionary storing prediction score for each instance
    :param query_roles: a list of roles for which the metric scores will be calculated
    :param role_to_values: a dictionary mapping from each role to candidate values
    :param ignore_non_object: ignore instances that do not represent a valid object, i.e., an instance that does not
                              have a valid class or does not have a valid value for the queried role
    :return:
    """
    # check predictions
    # NOTE: negative_data not included below bc I am only doing a
    # single role at a time; however, negative_data contains negatives
    # for all roles. Therefore, the check would always fail. Removing
    # negative_data should not affect performance
    #for instance in positive_data:# + negative_data:
    #    assert tuple(instance) in instance_prediction

    # save results and qualitative examples to file
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    results = {}
    for query_role in query_roles:

        # storing some stats
        # storing the rank of each positive instance
        query_ranks = []
        # number of negative instances generated for each positive instance
        num_negative_instances = []
        # number of roles that are the query role
        num_query_role = []
        # number of testing instance
        num_positive_instances = 0

        # storing each instance as a tuple of (groundtruth label, score, instance)
        test_instance_to_perturbed_instances = {}
        candidate_values = role_to_values[query_role]
        for instance in positive_data:
            # important: there may be multiple roles that are the query role in the instance. For example, an object
            #            can have more than one colors
            # position(s) of the query role in this positive instance
            query_role_pos = []
            object_class = None
            for i, rv_pair in enumerate(instance):
                if rv_pair[0] == query_role:
                    query_role_pos.append(i)
                if rv_pair[0] == "class":
                    object_class = rv_pair[1]

            if len(query_role_pos) == 0:
                continue

            if ignore_non_object:
                if object_class is None or object_class not in role_to_values["class"]:
                    continue

            # generate negative instances by perturbing the positive instance
            perturbed_instances = set()
            for role_i in query_role_pos:
                for perturbed_value in candidate_values:
                    perturbed_instance = copy.deepcopy(instance)
                    perturbed_instance[role_i] = (query_role, perturbed_value)
                    perturbed_instance = sorted(perturbed_instance)
                    # Important: the following is used to filter out instances that are positive instances in
                    #            test, train, or valid set
                    if perturbed_instance in negative_data:
                        perturbed_instance_tuple = tuple(sorted(perturbed_instance))
                        perturbed_instances.add(perturbed_instance_tuple)
            if len(perturbed_instances) == 0:
                continue

            instance_tuple = tuple(instance)
            instance_score = instance_prediction[instance_tuple]
            instance_scores = [instance_score]
            test_instance_to_perturbed_instances[instance_tuple] = [(1, instance_score, list(instance_tuple))]
            for perturbed_instance_tuple in perturbed_instances:
                instance_score = instance_prediction[perturbed_instance_tuple]
                instance_scores.append(instance_score)
                test_instance_to_perturbed_instances[instance_tuple].append((0, instance_score, list(perturbed_instance_tuple)))

            if None in instance_scores:
                print("Warning: None appears in predicted scores")
                print("Test positive instance: {}".format(instance))
                print("Query role: {}".format(query_role))
                for perturbed_instance_tuple in perturbed_instances:
                    print("Perturbed negative instance: {}".format(perturbed_instance_tuple,
                                                                   instance_prediction[perturbed_instance_tuple]))

            # the numpy method below does not properly deal with ties
            # instance_rank = np.where(np.argsort(instance_scores)[::-1] == 0)[0][0] + 1
            instance_rank = len(instance_scores) + 1 - rankdata(instance_scores, method='max')[0]
            query_ranks.append(instance_rank)
            num_positive_instances += 1
            num_negative_instances.append(len(perturbed_instances))
            num_query_role.append(len(query_role_pos))

        query_ranks = np.array(query_ranks)
        results[query_role] = {"MR": np.mean(query_ranks),
                               "MRR": np.mean(1.0 / query_ranks),
                               "Hit@1": sum(query_ranks <= 1) * 1.0 / len(query_ranks),
                               "Hit@2": sum(query_ranks <= 2) * 1.0 / len(query_ranks),
                               "Hit@3": sum(query_ranks <= 3) * 1.0 / len(query_ranks),
                               "Hit@5": sum(query_ranks <= 5) * 1.0 / len(query_ranks),
                               "Hit@10": sum(query_ranks <= 10) * 1.0 / len(query_ranks),
                               "# Tested Positive Instances": num_positive_instances,
                               "# Average Tested Negative Instances Per Positive Instance": np.mean(num_negative_instances),
                               "# Type-Constrained Candidate Value": len(role_to_values[query_role]),
                               "# Average Roles That Are Query Role In Each Positive Instance": np.mean(num_query_role)}

        if verbose:
            print("For query role {}".format(query_role))
            print("Mean Rank: {}".format(results[query_role]["MR"]))
            print("Mean Reciprocal Rank: {}".format(results[query_role]["MRR"]))
            print("Hit@1: {}".format(results[query_role]["Hit@1"]))
            print("Hit@2: {}".format(results[query_role]["Hit@2"]))
            print("Hit@3: {}".format(results[query_role]["Hit@3"]))
            print("Hit@5: {}".format(results[query_role]["Hit@5"]))
            print("Hit@10: {}".format(results[query_role]["Hit@10"]))
            print(
                "{} tested positive instances, {} perturbed negative instance for each positive instance on average".format(
                    results[query_role]["# Tested Positive Instances"],
                    results[query_role]["# Average Tested Negative Instances Per Positive Instance"]))
            print(
                "{} type-constrained candidate value, {} roles that are the query role for each positive instance on average".format(
                    results[query_role]["# Type-Constrained Candidate Value"],
                    results[query_role]["# Average Roles That Are Query Role In Each Positive Instance"]))

        # save results
        if save_dir:
            examples_for_role_file = os.path.join(save_dir, "{}:qualitative_examples.txt".format(query_role))
            with open(examples_for_role_file, "w") as fh:
                for key in test_instance_to_perturbed_instances:
                    all_test_instances = test_instance_to_perturbed_instances[key]
                    for label, score, instance in all_test_instances:
                        fh.write("{}\t{}\t{}".format(label, score, instance) + "\n")
                    fh.write("\n")

    df = pd.DataFrame(results)

    # compute average
    if len(query_roles) != 1:
        df['average'] = df.mean(axis=1)
        results = df.to_dict()

    if save_dir:
        if len(query_roles) == 1:
            results_file = os.path.join(save_dir, "{}:results.csv".format(query_role))
        else:
            results_file = os.path.join(save_dir, "results.csv")
        df.to_csv(results_file)

    return results


def mln_get_instance_examples_dict(positive_data, negative_data, query_role,
                                   role_to_values, ignore_non_object):
    """
    For the given query role, this function returns dict with key being the ID
    of the example (same as positive example index) and values of perturbed
    negative examples along with the positive example.

    :param positive_data: a list of positive examples
    :param negative_data: the complete list of negative examples. This list is used to check if a perturbed negative
    example is valid.
    :param query_role: the role that will be perturbed
    :param role_to_values: a dictionary mapping from each role to its candidate values
    :param ignore_non_object: whether to skip generating negative examples for object instance that doesn't have valid
    class category.
    :return: examples: a dictionary mapping from each positive instance ID to
    its corresponding examples (positives and negatives).
    """

    examples = {}
    candidate_values = role_to_values[query_role]
    idx = 0
    for instance in positive_data:
        # important: there may be multiple roles that are the query role in the instance. For example, an object
        #            can have more than one colors
        # position(s) of the query role in this positive instance
        query_role_pos = []
        object_class = None
        for i, rv_pair in enumerate(instance):
            if rv_pair[0] == query_role:
                query_role_pos.append(i)
            if rv_pair[0] == "class":
                object_class = rv_pair[1]

        if len(query_role_pos) == 0:
            continue

        if ignore_non_object:
            if object_class is None or object_class not in role_to_values["class"]:
                continue

        # generate negative instances by perturbing the positive instance
        perturbed_instances = set()
        for role_i in query_role_pos:
            for perturbed_value in candidate_values:
                perturbed_instance = copy.deepcopy(instance)
                perturbed_instance[role_i] = (query_role, perturbed_value)
                perturbed_instance = sorted(perturbed_instance)
                # Important: the following is used to filter out instances that are positive instances in
                #            test, train, or valid set
                if perturbed_instance in negative_data:
                    perturbed_instance_tuple = tuple(perturbed_instance)
                    perturbed_instances.add(perturbed_instance_tuple)

        instance_dict = {}
        values = []
        for role_i in query_role_pos:
            role, value = instance[role_i]
            values.append(value)
        instance_dict[''.join(sorted(values))] = {
            "rv": tuple(instance),
            "label": 1,
        }
        for perturbed_instance in perturbed_instances:
            values = []
            for role_i in query_role_pos:
                role, value = perturbed_instance[role_i]
                values.append(value)
            instance_dict[''.join(sorted(values))] = {
                "rv": perturbed_instance,
                "label": 0,
            }
        examples[idx] = instance_dict
        idx += 1

    return examples


def load_flattened_data(filename):
    flattened_data = []
    with open(filename, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                flattened_data.append(eval(line))
    return flattened_data


def format_instances_rv2atoms(instances):
    atoms = []
    for instance in instances:
        atom = []
        for rv in instance:
            role, value = rv
            atom.append(role + "(" + value + ")")
        atoms.append(atom)
    return atoms


def get_role_constraints(roles, instances):
    hard_roles = {}
    soft_roles = {}
    role_count_init = {}

    for role in roles.keys():
        hard_roles[role] = 1
        soft_roles[role] = 1
        role_count_init[role] = 0

    for instance in instances:
        role_count = copy.copy(role_count_init)
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
