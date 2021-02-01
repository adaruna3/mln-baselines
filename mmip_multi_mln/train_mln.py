from time import time
from argparse import ArgumentParser

from pracmln import MLN, MLNLearn
from pracmln.mln import Database

import pdb

if __name__ == "__main__":
    parser = ArgumentParser(description="Learn MLN")
    parser.add_argument("--input_mln", type=str, help="(.mln)", nargs="?",
                        default="./models/class_initial.mln")
    parser.add_argument("--input_database", type=str, help="(.db)", nargs="?",
                        default="./data/train.db")
    parser.add_argument("--output_mln", type=str, help="models", nargs="?",
                        default="./models/class_learned.mln")
    args = parser.parse_args()
    # loads the initial MLN and DBs
    mln = MLN.load(args.input_mln)
    dbs = Database.load(mln, args.input_database)
    # runs the learning on the markov logic network to get weights
    start = time()
    learned_mln = MLNLearn(mln=mln, db=dbs, verbose=True, method="BPLL_CG", use_prior=True, multicore=True).run()
    learned_mln.tofile(args.output_mln)
    duration = int( (time()-start) / 60.0)
    with open("./results/" + args.output_mln.split("/")[2].split(".")[0] + "_traintime.txt", "w") as f:
        f.write(" ---- %s minutes ---- \n" % duration)
