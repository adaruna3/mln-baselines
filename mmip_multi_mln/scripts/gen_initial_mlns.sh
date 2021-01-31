#!/bin/sh
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role class --output_mln ./models/class_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role color --output_mln ./models/color_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role dimension --output_mln ./models/dimension_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role purity --output_mln ./models/purity_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role dampness --output_mln ./models/dampness_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role room --output_mln ./models/room_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role temperature --output_mln ./models/temperature_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role price --output_mln ./models/price_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role material --output_mln ./models/material_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role weight --output_mln ./models/weight_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role physical_property --output_mln ./models/physical_property_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role shape --output_mln ./models/shape_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role specific_place --output_mln ./models/specific_place_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role transparency --output_mln ./models/transparency_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role size --output_mln ./models/size_initial.mln
python init_mln.py --formula_file ./data/formula_matrix_manual.mx --query_role spatial_distribution --output_mln ./models/spatial_distribution_initial.mln
