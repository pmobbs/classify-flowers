#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: 
# DATE CREATED:                                   
# REVISED DATE: 
# PURPOSE: Parse arguments
# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu
#
##
import argparse

def get_input_args_predict():

	parser = argparse.ArgumentParser()
	parser.add_argument('path_to_image', action='store', help='Image file to run prediction on')
	parser.add_argument('checkpoint', action='store', help='Model checkpoint file')
	parser.add_argument('--top_k', nargs='?', default='3', help='Return top KK most likely classes as --top_k default value 3')
	parser.add_argument('--category_names', nargs=1, default='', help='JSON file mapping categories to real names')
	parser.add_argument('--gpu', action="store_true", default=False, help='Use GPU for training as --gpu with default value False')

	args = parser.parse_args()
 
	return args
