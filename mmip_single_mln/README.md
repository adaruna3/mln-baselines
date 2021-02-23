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
Use the following commands to train and test MLN after generating role-value data following the top-level [README](https://github.com/wliu88/multimodal_interactive_perception/blob/master/README.md).
```bash
# move needed data over
cp ../../data/[CONFIG TYPE]/*.txt ./data
# initializes a new MLN
python init_mln.py
# generates a training database
python gen_db.py --output_database train.db --input_dataset train_data.txt
# trains the initialized MLN using the training database
python train_mln.py
# geneates a test database (see argparse for valiation)
python gen_db.py --output_database test.db --input_dataset test_data.txt
# test the trained MLN on the validation database
python test_mln.py --positive_database test.db --positive_dataset test_data.txt --negative_dataset test_data_negative.txt
```
