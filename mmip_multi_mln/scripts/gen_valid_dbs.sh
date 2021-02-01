#!/bin/sh
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/class_learned.mln                --output_database ./data/class_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/color_learned.mln                --output_database ./data/color_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/dimension_learned.mln            --output_database ./data/dimension_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/purity_learned.mln               --output_database ./data/purity_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/dampness_learned.mln             --output_database ./data/dampness_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/room_learned.mln                 --output_database ./data/room_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/temperature_learned.mln          --output_database ./data/temperature_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/price_learned.mln                --output_database ./data/price_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/material_learned.mln             --output_database ./data/material_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/weight_learned.mln               --output_database ./data/weight_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/physical_property_learned.mln    --output_database ./data/physical_property_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/shape_learned.mln                --output_database ./data/shape_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/specific_place_learned.mln       --output_database ./data/specific_place_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/transparency_learned.mln         --output_database ./data/transparency_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/size_learned.mln                 --output_database ./data/size_val.db
python gen_db.py --input_datasets ./data/val_data.txt --input_mln ./models/spatial_distribution_learned.mln --output_database ./data/spatial_distribution_val.db
