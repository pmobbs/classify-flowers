#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Paul Mobbs
# DATE CREATED: 12/12/18                                 
# REVISED DATE: 
# PURPOSE: 
# Train a new network on a data set with train.py
# 
# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu
#
##

# Imports python modules
from time import time, sleep

from get_input_args_train import get_input_args_train

# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # This function retrieves Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_input_args_train()

    

    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()
