#!/bin/sh
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/class_learned.mln                --output_database ./data/class_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/color_learned.mln                --output_database ./data/color_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/dimension_learned.mln            --output_database ./data/dimension_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/purity_learned.mln               --output_database ./data/purity_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/dampness_learned.mln             --output_database ./data/dampness_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/room_learned.mln                 --output_database ./data/room_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/temperature_learned.mln          --output_database ./data/temperature_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/price_learned.mln                --output_database ./data/price_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/material_learned.mln             --output_database ./data/material_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/weight_learned.mln               --output_database ./data/weight_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/physical_property_learned.mln    --output_database ./data/physical_property_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/shape_learned.mln                --output_database ./data/shape_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/specific_place_learned.mln       --output_database ./data/specific_place_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/transparency_learned.mln         --output_database ./data/transparency_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/size_learned.mln                 --output_database ./data/size_test.db
python gen_db.py --input_datasets ./data/test_data.txt --input_mln ./models/spatial_distribution_learned.mln --output_database ./data/spatial_distribution_test.db
