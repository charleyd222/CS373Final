import sqlite3
import pandas as pd
import numpy as np
from torch import nn
import torch
from makemidi import makeMidi

# some code from https://medium.com/@abhilashkrish/step-by-step-guide-to-music-generation-using-rnns-with-pytorch-2fbf1a4172a3

def find_instruments(ins, db = 'assets/wjazzd.db'): # Finds melids of solos of a certain instrument
    # access data from sq database
    con = sqlite3.connect(db)
    solo_info = pd.read_sql_query("SELECT * FROM solo_info", con)
    melody = pd.read_sql_query("SELECT * FROM melody", con)

    # Find all melid's with instrument
    melid = []
    avgtempo = []
    for i in range(456):
        if solo_info['instrument'][i] == str(ins):
            melid += [solo_info['melid'][i]]
            avgtempo += [solo_info['avgtempo'][i]]

    # Get data from each melid
    data = {}
    for i, m in enumerate(melid):
        data[i] = melody[melody['melid'] == m]
    return data, avgtempo

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
    data_duration = []
    data_step = []
    data_end = []

    # Get all instrument (default piano) data 
    df, bpms = find_instruments(instrument)
    for data_set in df: # iterate through each solo
        # pitch and duration
        # create contout
        # k = df[data_set]['duration'].keys()[0] # duration of first event
        # for i in range(len(df[data_set]['pitch'])): 
        #     data_duration += [df[data_set]['duration'][i + k]]
        #     data_pitch += [df[data_set]['pitch'][i + k]]
        #     if i > (contour_len - 1):
        #         data_contour += [np.average(
        #             df[data_set]['pitch'][(i - contour_len):i])]
        #     else:
        #         data_contour += [0]
        step = 0
        for i in range(len(df[data_set]['pitch'])): # count for each pitch in the df
            if i == len(df[data_set]['pitch']) - 1: # at the end
                data_end.append(1)
            else:
                data_end.append(0)
            data_duration.append(df[data_set]['duration'].iloc[i]) # add duration of current event
            data_pitch.append(df[data_set]['pitch'].iloc[i]) # add pitch of current event
            data_step.append(step)
            if i < len(df[data_set]['pitch']) - 1:
                step = df[data_set]['onset'].iloc[i + 1] - df[data_set]['onset'].iloc[i] - df[data_set]['duration'].iloc[i] # length of silence after current note
    
    # Convert data to either pandas df or to torch tensor
    df = np.array([data_pitch, data_step, data_duration, data_end]).T
    if pdf:
        df = pd.DataFrame(df, columns=['pitch','step','duration', 'end'])
    else:
        df = torch.from_numpy(df).float()

    return df

def make_sequences(df, sequence_length=16):
    #print(df.shape)
    num_sequences = len(df) - sequence_length # find the number of overlapping sequences
    # X = np.empty((num_sequences, 4))
    # y = np.empty((num_sequences, sequence_length, 4))
    
    X = []
    y = []
    for i in range(num_sequences):
        X.append(df[i:i+sequence_length]) # add the sequence as the data
        y.append(df[i+sequence_length]) # add the label
    return X, y
    

class rnn_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(rnn_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        #print(f"x,shape: {x.shape}")
        #print(f"hidden,shape: {hidden[0].shape}")
        out, hidden = self.lstm(x, hidden)
        #print(f"out: {out}")
        # print(out.shape)
        # print(out)
        # h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        # out, _ = self.rnn(x, h0)
        # out = self.fc(out[:, -1, :])
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self):
        #print(self.num_layers, self.hidden_size)
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        #print(f'h0: {h0.shape}')
        #print(f'c0: {c0.shape}')
        #print(h0)
        return (h0, c0)

def train_model(X, y, model, rate, epochs):
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Training loop
    
    for epoch in range(epochs):
        n = 0
        for Xi, yi in zip(X, y):
            #print(Xi, yi)
            #print(Xi.shape)
            # model.train()
            # outputs = model(X.unsqueeze(2))  # Add a dimension for input size
            # loss = loss_func(outputs, y.unsqueeze(2))
            
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            Xi = Xi.unsqueeze(0).to(device)
            yi = yi.to(device)
            hidden = model.init_hidden()
            outputs, hidden = model(Xi, hidden)
            #print(f'output: {outputs.shape}')
            #print(f'yi: {yi.shape}')
            #print(outputs, yi)
            loss = loss_func(outputs[0,0,:], yi)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n+= 1
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Total runs: {n}')

def test_model(model, start_sequence, max_length):
    model.eval()
    input_seq = torch.tensor(start_sequence, dtype=torch.float32).unsqueeze(0)
    generated_sequence = start_sequence
    hidden = model.init_hidden()
    
    for _ in range(max_length):
        output, hidden = model(input_seq, hidden)
        #_, predicted_event = torch.max(output.data, 1) # choose most probable next event
        predicted_event = output.detach().numpy()[0,:1,:]
        print(predicted_event.shape)
        #print(predicted_event[0]) # get data from tensor
        print(generated_sequence.shape)
        generated_sequence = np.concat((generated_sequence, predicted_event), 0)
        #print(generated_sequence.shape) # add predicted event to sequence
        input_seq = torch.cat((input_seq, torch.tensor(predicted_event).unsqueeze(1)), dim = 1) # add predicted event to next input

        #print(input_seq.shape)
        input_seq = input_seq[:,1:,:]
    
    return generated_sequence        

if __name__ == "__main__":
    d = create_data()
    
    input_size = 4
    hidden_size = 128
    output_size = 4
    rate = 0.01
    epochs = 100
    sequence_length = 16
    max_length = 100 # number of new events to generate in composition
    X, y = make_sequences(d, sequence_length)
    model = rnn_model(input_size, hidden_size, output_size, 1)

    train_model(X, y, model, rate, epochs)
    
    #start_sequence = X[0] # take first sequence of training set as test
    #sequence = test_model(model, start_sequence, max_length)
    #print(sequence)
    # print(predictions)
    # makeMidi(predictions)

    import pickle 
    
    # Open a file and use dump() 
    with open('model.pkl', 'wb') as file: 
        pickle.dump(model, file) 