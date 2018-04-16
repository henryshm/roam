# ROAM: A Neighborhood Explorer using t-SNE

Please see the LICENSE.

## Overview

![ROAM Screenshot](roam-image.png)

## Getting Started

Roam is written for Python 3.

In addition, you will need the following packages installed:

- numpy
- scipy
- scikit-learn
- pandas
- matplotlib
- bottle
- requests
- munch

## Running Roam 

Once you have installed the necessary dependencies, you can start the server and client programs using the following commands.

    cd src
    
    # run the server process
    ./roamserver.py ../data/automobiles.csv & 
    
    # run the GUI (uses server)
    ./roamgui.py
