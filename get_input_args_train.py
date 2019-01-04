#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: 
# DATE CREATED:                                   
# REVISED DATE: 
# PURPOSE: Parse arguments
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu
#
##
import argparse

def get_input_args_train():

	parser = argparse.ArgumentParser()
	parser.add_argument('--arch', nargs='?', default='vgg13', help='Choose architecture as --arch \'vgg13\' or \'vgg16\' with default value \'vgg13\'')
	parser.add_argument('--learning_rate', nargs='?', default='0.01', help='Set learning rate hyperparameter as --learning_rate with default value 0.01')
	parser.add_argument('--hidden_units', nargs='?', default='512', help='Set hidden units hyperparameter as --hidden_units with default value 512')
	parser.add_argument('--epochs', nargs='?', default='2', help='Set epochs hyperparameter as --epochs with default value 2')
	parser.add_argument('--gpu', nargs='?', default='0', help='Use GPU for training as --gpu with default value 0')

	args = parser.parse_args()

	return args
