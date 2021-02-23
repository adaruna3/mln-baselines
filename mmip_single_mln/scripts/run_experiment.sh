#!/bin/sh
python init_mln.py
python gen_db.py --output_database train.db --input_dataset train_data.txt
python gen_db.py --output_database test.db --input_dataset test_data.txt
python train_mln.py
python test_mln.py --positive_database test.db --positive_dataset test_data.txt --negative_dataset test_data_negative.txt
