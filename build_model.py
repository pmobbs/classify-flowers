#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Paul Mobbs
# DATE CREATED: 1/3/19                                 
# REVISED DATE: 
# PURPOSE: 
# Build model based on provided parameters
# 
##

# Imports python modules
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models

def build_model(arch, num_labels, hidden_units):
    if (arch == 'vgg13'):
        print("Instantiating vgg13 model")
        model = models.vgg13(pretrained=True)
    else:
        print("Instantiating vgg16 model")
        model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    features = list(model.classifier.children())[:-1]

    # number of filters in the bottleneck layer
    num_filters = model.classifier[len(features)].in_features

    # add layers
    features.extend([
        nn.Dropout(),
        nn.Linear(num_filters, hidden_units),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(True),
        nn.Linear(hidden_units, num_labels),
    ])
    model.classifier = nn.Sequential(*features)

    return model