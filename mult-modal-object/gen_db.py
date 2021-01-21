from pracmln import MLN
from pracmln.mln import Database

import pdb


def load_flattened_data(filename):
    flattened_data = []
    with open(filename, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                flattened_data.append(eval(line))
    return flattened_data


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
    # loads data for DBs
    tr = load_flattened_data("train_data.txt")
    # va = load_flattened_data("val_data.txt")
    va = []
    # loads the initial MLN
    mln = MLN.load("initial.mln")
    # format from role-value to atoms and save as MLN DBs
    atoms = format_rv2atoms(tr+va)
    dbs = generate_databases(mln, atoms)
    with open("train.db", "w") as f:
        Database.write_dbs(dbs, f)
    print("The training database for the MLN is in 'train.db'.")