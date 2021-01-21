from pracmln import MLN, MLNLearn
from pracmln.mln import Database
import time

import pdb

if __name__ == "__main__":
    # loads the initial MLN and DBs
    mln = MLN.load("initial.mln")
    dbs = Database.load(mln, "train.db")
    # runs the learning on the markov logic network to get weights
    start_time = time.time()
    learned_mln = MLNLearn(mln=mln, db=dbs, verbose=True, method="BPLL_CG", use_prior=True, multicore=True).run()
    learned_mln.tofile("learned.mln")
    print(" ---- %s seconds ---- " % (time.time() - start_time))