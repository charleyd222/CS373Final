import sqlite3
import pandas as pd
import numpy as np
from torch import nn
import torch
from makemidi import makeMidi
from dataLoad import test_model, rnn_model, create_data, make_sequences
import pickle

d = create_data()
    
sequence_length = 16
X, y = make_sequences(d, sequence_length)

# Open the file in binary mode 
with open('model.pkl', 'rb') as file: 
    model = pickle.load(file) 

start_sequence = X[0].numpy() # take first sequence of training set as test
sequence = test_model(model, start_sequence, 50)
print(sequence)
# print(predictions)
makeMidi(sequence)