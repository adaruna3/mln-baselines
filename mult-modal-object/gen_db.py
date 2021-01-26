from argparse import ArgumentParser
import json

from pracmln import MLN
from pracmln.mln import Database
from data_utils import load_flattened_data, format_instances_rv2atoms

import pdb


def generate_databases(mln, instances):
    dbs = []
    for atoms in instances:
        db = Database(mln)
        for atom in atoms:
            db[atom] = 1.0
        dbs.append(db)
    return dbs


def rvs2mlnrvs(mln, roles, instances):
    all_roles = set(roles.keys())
    for instance in instances:
        instance_roles = set()
        for role, value in instance:
            instance_roles.add(role)
        missing = all_roles.difference(instance_roles)
        for role in missing:
            instance.append(tuple((role, "None")))
    pdb.set_trace()


if __name__ == "__main__":
    parser = ArgumentParser(description="Role-Value Dataset 2 MLN Database")
    parser.add_argument('input_mln', type=str, help='(.mln)')
    parser.add_argument('input_dataset', type=str, help='(.txt)')
    parser.add_argument('output_database', type=str, help='(.db)')
    args = parser.parse_args()

    # loads the initial MLN
    mln = MLN.load(args.input_mln)
    # loads data for DBs
    with open("role_to_values.json", "r") as f:
        roles = json.loads(f.readlines()[0])
    rv = load_flattened_data(args.input_dataset)
    rv = rvs2mlnrvs(mln, roles, rv)
    # format from role-value to atoms and save as MLN DBs
    atoms = format_instances_rv2atoms(rv)
    dbs = generate_databases(mln, atoms)
    with open(args.output_database, "w") as f:
        Database.write_dbs(dbs, f)
    print("The database for the MLN is in " + args.output_database + ".")
