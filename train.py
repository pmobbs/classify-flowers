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

from get_input_args_train import get_input_args_train
from build_model import build_model

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def train(model, trainloader, criterion, optimizer, valloader, epochs, device, arch, num_classes, hidden_units):
    if (epochs < 1):
        epochs = 1

    start_epoch = 0
    steps = 0
    running_loss = 0
    print_every = 10
    start_time = time()

    for e in range(epochs):
        model.train()
        
        for images, labels in trainloader:        
            steps += 1
            
            if (e < start_epoch):
                # Skip images for this epoch
                continue

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, valloader, criterion, device)
                    
                end_time = time()
                tot_time = end_time - start_time
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    str(int(100*(steps-e*len(trainloader))/len(trainloader))) + "% done ",
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss/len(valloader)),
                    "Test Accuracy: {:.3f}".format(accuracy/len(valloader)),
                    "Total Elapsed Runtime:",
                    str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
                    +str(int((tot_time%3600)%60)))
                
                running_loss = 0
                
                # Make sure training is back on
                model.train()

        if (e < start_epoch):
            print("Skip writing for epoch "+str(e+1))
            continue
        
        save_checkpoint('checkpoint'+str(e%2)+'.pth', e, arch, num_classes, hidden_units, model.state_dict())
        
def save_checkpoint(filename, e, arch, num_classes, hidden_units, state_dict):
        # save the state at the end of each epoch
        print('Writing model to: '+filename)
        checkpoint_state = {'epoch': e,
            'arch': arch,
            'num_classes': num_classes,
            'hidden_units': hidden_units,
            'state_dict': state_dict}
        torch.save(checkpoint_state, filename)

# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # Default locations for training data
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # This function retrieves Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_input_args_train()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and in_arg.gpu == '1' else "cpu")
    
    print("arch=\'" + in_arg.arch + "\', epochs=\'" + in_arg.epochs + "\', gpu=\'" + in_arg.gpu + "\', " + 
          "hidden_units=\'" + in_arg.hidden_units + "\', learning_rate=\'" + in_arg.learning_rate + "\'")

    # Define transformations on training and validation data
    data_size = 224
    data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(data_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    data_transforms_test = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(data_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    val_data = datasets.ImageFolder(valid_dir, transform=data_transforms_test)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms_test)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    print("Loaded dataset size for training: " + str(len(train_data)))
    print("Loaded dataset size for validation: " + str(len(val_data)))
    print("Loaded dataset size for testing: " + str(len(test_data)))

    print("Classes: ")
    class_names = train_data.classes
    num_labels = len(class_names)
    print(str(num_labels))    

    hidden_units = int(in_arg.hidden_units)
    
    # build model based on specified parameters
    model = build_model(in_arg.arch, num_labels, hidden_units)
        
    # Define optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.classifier.parameters()), lr=float(in_arg.learning_rate))

    # Move to GPU (if available)
    model.to(device)

    # Do training
    #print("Starting training...")
    #train(model, trainloader, criterion, optimizer, valloader, int(in_arg.epochs), device, in_arg.arch, num_labels, hidden_units)

    # Write final checkpoint
    save_checkpoint('checkpoint.pth', int(in_arg.epochs), in_arg.arch, num_labels, hidden_units, model.state_dict())

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
