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
Use the following commands to train and test MLN after copying ove role-value data into ./data
```bash
# initializes a new MLN
python init_mln.py
# generates a training database
python gen_db.py
# trains the initialized MLN using the training database
python learn_mln.py
# geneates a validation database (see argparse for testing)  
python gen_db.py --input_datasets ./data/val_data.txt --output_database ./data/valid.db
# test the trained MLN on the validation database
python test_mln.py
```
