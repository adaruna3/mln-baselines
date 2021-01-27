from copy import deepcopy
from argparse import ArgumentParser

from pracmln import MLN
from pracmln.mln import Database
import data_utils as utils

import pdb


def generate_databases(mln, instances):
    dbs = []
    for atoms in instances:
        db = Database(mln)
        for atom in atoms:
            db[atom] = 1.0
        dbs.append(db)
    return dbs


def rvs2mlnrvs(roles, instances):
    mlninstances = deepcopy(instances)
    all_roles = set(roles.keys())
    mv_roles = [r for r, c in utils.get_role_constraints(roles, mlninstances).items() if c == '']
    for instance in mlninstances:
        instance_roles = set()
        instance_mv_roles = {role: [] for role in mv_roles}
        rv_idx = 0
        while rv_idx < len(instance):
            role, value = instance[rv_idx]
            instance_roles.add(role)
            if role in mv_roles:
                instance_mv_roles[role].append(value)
                del instance[rv_idx]
            else:
                rv_idx += 1
        missing = all_roles.difference(instance_roles)
        for role in missing:
            instance.append(tuple((role, "None")))
            if role in instance_mv_roles:
                del instance_mv_roles[role]
        for role, values in instance_mv_roles.items():
            values = ''.join(sorted(values))
            instance.append(tuple((role, values)))
    return mlninstances


if __name__ == "__main__":
    parser = ArgumentParser(description="Role-Value Dataset 2 MLN Database")
    parser.add_argument("--input_mln", type=str, help="(.mln)", nargs="?",
                        default="./models/initial.mln")
    parser.add_argument("--input_datasets", type=str, help="(.txt)", nargs="*",
                        default=["./data/train_data.txt"])
    parser.add_argument("--output_database", type=str, help="(.db)", nargs="?",
                        default="./data/train.db")
    parser.add_argument("--roles_file", type=str, help="(.txt)", nargs="?",
                        default="./data/role_to_values.txt")
    args = parser.parse_args()
    # loads the initial MLN
    mln = MLN.load(args.input_mln)
    # loads data for DBs
    atoms = []
    with open(args.roles_file, "r") as f:
        roles = eval(f.readlines()[0])
    for input_dataset in args.input_datasets:
        rv = utils.load_flattened_data(input_dataset)
        rv = rvs2mlnrvs(roles, rv)
        # formats role-value to atoms
        atoms += utils.format_instances_rv2atoms(rv)
    # generates the DBs and saves
    dbs = generate_databases(mln, atoms)
    with open(args.output_database, "w") as f:
        Database.write_dbs(dbs, f)
    print("The database for the MLN is in " + args.output_database + ".")
