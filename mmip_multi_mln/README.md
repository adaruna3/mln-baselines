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
pip install pracmln numpy pandas scipy
```

## Quickstart
Use the following commands to train and test MLNs after generating role-value data following the top-level [README](https://github.com/wliu88/multimodal_interactive_perception/blob/master/README.md).
```bash
# move needed data over
cp ../../data/[CONFIG TYPE]/*.txt ./data
# initializes an MLN for each role
./scripts/gen_initial_mlns.sh
# generates the single training database for all MLNs
python gen_db.py
# trains each initialized MLN using the training database
./scripts/train_mlns.sh
# geneates a validation database (see other script testing)
./scripts/gen_valid_dbs.sh
# test the trained MLN on the validation database
./scripts/validate_mlns.sh
```
