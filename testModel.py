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

p = {'46':0, '48':0, '50':0, '52':0, '54':0, '56':0, '58':0, '60':0, '62':0, '64':0, '66':0, '68':0, '70':0, '72':0, '74':0, '76':0, '78':0,'80':0}
#p = {'46':0, '48':0, '50':0, '52':0, '54':0, '55':0, '56':0, '57':0, '58':0, '59':0, '60':0, '61':0, '62':0, '63':0, '64':0, '65':0, '66':0, '67':0, '68':0, '69':0, '70':0, '72':0, '74':0, '76':0, '78':0,'80':0}
#p = {'46':0, '47':0, '48':0, '49':0, '50':0, '51':0, '52':0, '53':0, '54':0, '55':0, '56':0, '57':0, '58':0, '59':0, '60':0, '61':0, '62':0, '63':0, '64':0, '65':0, '66':0, '67':0, '68':0, '69':0, '70':0, '71':0, '72':0, '73':0, '74':0, '75':0, '76':0, '77':0, '78':0, '79':0,'80':0}

p_inv = {}
for i, n in enumerate(p):
    p[n] = i
    p_inv[i] = n
print(p_inv[7])
start_sequence = X[0].numpy() # take first sequence of training set as test
sequence = test_model(model, start_sequence, 50)
print(sequence.shape)
pitch_sequence = []
for i in sequence:
    pitch_sequence += [p_inv[int(i.argmax())]]

print(pitch_sequence)
makeMidi(pitch_sequence)