#!/bin/sh
python init_mln.py
python gen_db.py initial.mln train.db --input_dataset train_data.txt val_data.txt
python gen_db.py initial.mln test.db --input_dataset test_data.txt
python learn_mln.py initial.mln train.db learned.mln
python test_mln.py learned.mln valid.db val_data.txt val_data_negative.txt
