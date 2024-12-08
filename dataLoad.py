import sqlite3
import pandas as pd
import numpy as np
from torch import nn
import torch
from makemidi import makeMidi
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

def create_data_bin(bin_size, instrument = 'p', contour_len = 4, pdf = False):
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

def create_data(instrument = 'p', contour_len = 4, pdf = False):
    data_pitch = []
    data_contour = []
    data_duration = []

    # Get all instrument (default piano) data 
    df = find_instruments(instrument)

    for data_set in df:
        # pitch and duration
        

        # create contout
        k = df[data_set]['duration'].keys()[0]
        for i in range(len(df[data_set]['pitch'])):
            data_duration += [df[data_set]['duration'][i + k]]
            data_pitch += [df[data_set]['pitch'][i + k]]
            if i > (contour_len - 1):
                data_contour += [np.average(
                    df[data_set]['pitch'][(i - contour_len):i])]
            else:
                data_contour += [0]
    print(data_duration[:5])
    # Convert data to either pandas df or to torch tensor
    df = np.array([data_pitch, data_contour, data_duration]).T
    if pdf:
        df = pd.DataFrame(df, columns=['pitch','contour'])
    else:
        df = torch.from_numpy(df).float()

    return df

class rnn_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(rnn_model, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

def train_model(X, y, model):
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X.unsqueeze(2))  # Add a dimension for input size
        loss = loss_func(outputs, y.unsqueeze(2))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    d = create_data()
    #print(d[:20])
    X = d[:-1]
    y = d[1:]
    
    input_size = 1
    hidden_size = 200
    output_size = 2
    rate = 0.001
    model = rnn_model(input_size, hidden_size, output_size)
    print(X.shape)

    train_model(X, y, model)

    model.eval()
    with torch.no_grad():
        predictions = model(X.unsqueeze(2)).squeeze(2).numpy()

    print(predictions)

    #makeMidi(predictions)

    
