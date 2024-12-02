import sqlite3
import pandas as pd
import numpy as np
from torch import nn
import torch

def find_instruments(ins, db = 'assets/wjazzd.db'): # Finds melids of solos of a certain instrument
    # access data from sq database
    con = sqlite3.connect(db)
    solo_info = pd.read_sql_query("SELECT * FROM solo_info", con)
    melody = pd.read_sql_query("SELECT * FROM melody", con)

    # Find all melid's with instrument
    melid = []
    for i in range(456):
        if solo_info['instrument'][i] == str(ins):
            melid += [solo_info['melid'][i]]

    # Get data from each melid
    data = {}
    for i, m in enumerate(melid):
        data[i] = melody[melody['melid'] == m]

    return data

def create_data(bin_size, instrument = 'p', contour_len = 4, pdf = False):
    data_pitch = []
    data_contour = []

    # Get all instrument (default piano) data 
    df = find_instruments(instrument)

    for data_set in df:
        #initial data key for data_set
        k = df[data_set]['pitch'].keys()[0]
        k0 = k

        #Final data key
        kf = df[data_set]['pitch'].keys()[-1]
        
        # time and duration for binning
        t = 0
        dur = df[data_set]['duration'][k]
        tot_dur = 0

        data_pitch_temp = []
        data_contour_temp = []
        
        while k < kf + 1: # while theres still more keys
            # duration of instance
            dur = df[data_set]['duration'][k]

            # add pitch of instance
            data_pitch += [df[data_set]['pitch'][k]]

            # create contout
            if k - k0 > (contour_len - 1):
                data_contour += [np.average(
                    df[data_set]['pitch'][(k - k0 - contour_len):(k - k0)])]
            else:
                data_contour += [0]
            
            # increase time by bin size
            t += bin_size

            # if we have gone over the duration of the current instance, move on to next
            if t >  dur:
                k += 1
                t = 0

    # Convert data to either pandas df or to torch tensor
    df = np.array([data_pitch, data_contour]).T
    if pdf:
        df = pd.DataFrame(df, columns=['pitch','contour'])
    else:
        df = torch.from_numpy(df).float()

    return df

if __name__ == "__main__":
    d = create_data(0.01)
    print(d[:20])
    rnn = nn.RNN(2, 20)
    output, hn = rnn(d)

    print(output)
