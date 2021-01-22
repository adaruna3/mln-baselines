import time
from argparse import ArgumentParser

from pracmln import MLN, MLNLearn
from pracmln.mln import Database

import pdb

if __name__ == "__main__":
    parser = ArgumentParser(description="Learn MLN")
    parser.add_argument('input_mln', type=str, help='(.mln)')
    parser.add_argument('input_database', type=str, help='(.db)')
    parser.add_argument('output_mln', type=str, help='(.mln)')
    args = parser.parse_args()

    # loads the initial MLN and DBs
    mln = MLN.load(args.input_mln)
    dbs = Database.load(mln, args.input_database)
    # runs the learning on the markov logic network to get weights
    start_time = time.time()
    learned_mln = MLNLearn(mln=mln, db=dbs, verbose=True, method="BPLL_CG", use_prior=True, multicore=True).run()
    learned_mln.tofile(args.output_mln)
    print(" ---- %s seconds ---- " % (time.time() - start_time))