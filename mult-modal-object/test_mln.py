from pracmln import MLN, MLNQuery
from pracmln.mln import Database


if __name__ == "__main__":
    learned_result = MLN.load("learned_weights.mln")
    MLNQuery(queries="OperatesOn(sink-l,carrot-o)", verbose=True, mln=learned_result, db=db).run()