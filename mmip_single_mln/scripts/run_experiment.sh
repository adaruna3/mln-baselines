#!/bin/sh
python init_mln.py
python gen_db.py --output_database ./data/train.db --input_dataset ./data/train_data.txt
python gen_db.py --output_database ./data/test.db --input_dataset ./data/test_data.txt
python train_mln.py
python test_mln.py --positive_database ./data/test.db --positive_dataset ./data/test_data.txt --negative_dataset ./data/test_data_negative.txt
