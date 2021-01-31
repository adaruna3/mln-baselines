#!/bin/sh
python train_mln.py --input_mln ./models/class_initial.mln                --output_mln ./models/class_learned.mln
python train_mln.py --input_mln ./models/color_initial.mln                --output_mln ./models/color_learned.mln
python train_mln.py --input_mln ./models/dimension_initial.mln            --output_mln ./models/dimension_learned.mln
python train_mln.py --input_mln ./models/purity_initial.mln               --output_mln ./models/purity_learned.mln
python train_mln.py --input_mln ./models/dampness_initial.mln             --output_mln ./models/dampness_learned.mln
python train_mln.py --input_mln ./models/room_initial.mln                 --output_mln ./models/room_learned.mln
python train_mln.py --input_mln ./models/temperature_initial.mln          --output_mln ./models/temperature_learned.mln
python train_mln.py --input_mln ./models/price_initial.mln                --output_mln ./models/price_learned.mln
python train_mln.py --input_mln ./models/material_initial.mln             --output_mln ./models/material_learned.mln
python train_mln.py --input_mln ./models/weight_initial.mln               --output_mln ./models/weight_learned.mln
python train_mln.py --input_mln ./models/physical_property_initial.mln    --output_mln ./models/physical_property_learned.mln
python train_mln.py --input_mln ./models/shape_initial.mln                --output_mln ./models/shape_learned.mln
python train_mln.py --input_mln ./models/specific_place_initial.mln       --output_mln ./models/specific_place_learned.mln
python train_mln.py --input_mln ./models/transparency_initial.mln         --output_mln ./models/transparency_learned.mln
python train_mln.py --input_mln ./models/size_initial.mln                 --output_mln ./models/size_learned.mln
python train_mln.py --input_mln ./models/spatial_distribution_initial.mln --output_mln ./models/spatial_distribution_learned.mln
