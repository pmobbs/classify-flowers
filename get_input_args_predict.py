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
	parser.add_argument('--top_k', nargs='?', default='3', help='Return top KK most likely classes as --top_k default value 3')
	parser.add_argument('--category_names', nargs='?', default='', help='Use a mapping of categories to real names as --category_names with no default value')
	parser.add_argument('--gpu', nargs='?', default='0', help='Use GPU for inference as --gpu with default value 0')

	args = parser.parse_args()
 
	return args
