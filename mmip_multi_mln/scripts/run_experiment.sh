#!/bin/sh
./scripts/gen_initial_mlns.sh
python gen_db.py
./scripts/train_mlns.sh
./scripts/gen_test_dbs.sh 
./scripts/test_mlns.sh
