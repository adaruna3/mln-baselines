from argparse import ArgumentParser

from pracmln import MLN
from pracmln.mln import Database
from data_utils import load_flattened_data

import pdb


def format_rv2atoms(instances):
    atoms = []
    for instance in instances:
        atom = []
        for rv in instance:
            role, value = rv
            atom.append(role + "(" + value + ")")
        atoms.append(atom)
    return atoms


def generate_databases(mln, instances):
    dbs = []
    for atoms in instances:
        db = Database(mln)
        for atom in atoms:
            db << atom
        dbs.append(db)
    return dbs


if __name__ == "__main__":
    parser = ArgumentParser(description="Role-Value Dataset 2 MLN Database")
    parser.add_argument('input_mln', type=str, help='(.mln)')
    parser.add_argument('input_dataset', type=str, help='(.txt)')
    parser.add_argument('output_database', type=str, help='(.db)')
    args = parser.parse_args()

    # loads data for DBs
    rv = load_flattened_data(args.input_dataset)
    # loads the initial MLN
    mln = MLN.load(args.input_mln)
    # format from role-value to atoms and save as MLN DBs
    atoms = format_rv2atoms(rv)
    dbs = generate_databases(mln, atoms)
    with open(args.output_database, "w") as f:
        Database.write_dbs(dbs, f)
    print("The database for the MLN is in " + args.output_database + ".")
