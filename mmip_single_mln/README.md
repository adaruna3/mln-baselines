# MLN Baseline - each file below has command parse to help
* Make MLN structure/formulas with "init_mln.py"
* Generate the training and testing MLN databases with "gen_db.py"
* Learn the weights of the MLN formulas with "learn_mln.py"
* Test the learned MLN with "test_mln.py", metrics from "data_utils.py"

## Setup
Use the following commands to install MLN pacakge
```bash
conda create -n pracmln python=3.6
conda activate pracmln
pip install pracmln
```

## Quickstart
Use the following commands to train and test MLN
```bash
python init_mln.py
python gen_db.py initial.mln train_data.txt train.db
python learn_mln.py initial.mln train.db trained.mln
python gen_db.py initial.mln test_data.txt test.db
python test_mln.py trained.mln test.db test_data.txt test_data_negative.txt
```
