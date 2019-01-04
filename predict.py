#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Paul Mobbs
# DATE CREATED: 12/12/18                                 
# REVISED DATE: 
# PURPOSE: 
# Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
# 
# Basic usage: python predict.py /path/to/image checkpoint
# Options:
# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu
#
##

# Imports python modules
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
import time
import torch.optim as optim


import json
from PIL import Image
import numpy as np
from torch.autograd import Variable
import matplotlib.image as mpimg

from time import time, sleep

from get_input_args_predict import get_input_args_predict
from build_model import build_model

def load_checkpoint(filename):
    state_dict = torch.load(filename)
    print("Loading checkpoint:")
    print("->Epochs: " + str(state_dict.get('epoch')))
    print("->Arch: " + str(state_dict.get('arch')))
    print("->Classes: " + str(state_dict.get('num_classes')))
    print("->Hidden Units: " + str(state_dict.get('hidden_units')))
    
    if (not(state_dict.get('arch') == 'vgg13' or state_dict.get('arch') == 'vgg16')):
        print('Model has unknown architecture.')
        return(1)

    if (state_dict.get('num_classes') == '' or int(state_dict.get('num_classes')) < 1):
        print('Model has invalid number of classes.')
        return(1)

    arch = state_dict.get('arch')
    hidden_units = state_dict.get('hidden_units')
    num_labels = state_dict.get('num_classes')

    # build model based on loaded parameters
    model = build_model(arch, num_labels, hidden_units)
    
    # load stored values into network
    model.load_state_dict(state_dict.get('state_dict'))
    
    return model

def predict(image_path, model, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img_pil = Image.open(image_path)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # preprocess the image
    img_tensor = preprocess(img_pil)

    img_tensor.unsqueeze_(0)
    
    img_tensor.requires_grad_(False)
    model.to(device)
    model = model.eval()
    output = model(img_tensor.to(device))
    output = output.cpu()
    
    return output.topk(topk)[0].data.numpy()[0], output.topk(topk)[1].data.numpy()[0]

# Lookup labels for list of classes in file
def classes_to_labels(classes, category_names):
    # Load labels
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    labels = {int(key):value for (key, value) in cat_to_name.items()}
    
    class_labels = []
    for i in classes:
        try:
            val = labels[i]
        except:
            val = "Unknown:"+str(i)
        class_labels.append(val)    

    return class_labels

# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # This function retrieves Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_input_args_predict()

    device = torch.device("cuda:0" if torch.cuda.is_available() and in_arg.gpu == '1' else "cpu")

    # Load checkpoint
    model = load_checkpoint(in_arg.model[0])
    
    # Run prediction
    probs, classes = predict(in_arg.image[0],model, int(in_arg.top_k), device)
    print("Top-K Probabilities: " + str(probs))
    print("Classes: " + str(classes))
    
    if (in_arg.category_names != ''):
        print("Labels:" + str(classes_to_labels(classes, in_arg.category_names[0])))
    
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
