from sys import argv
import sqlite3
import pandas as pd
import numpy as np
from torch import nn
import torch
from makemidi import makeMidi
import pickle

# some code from https://medium.com/@abhilashkrish/step-by-step-guide-to-music-generation-using-rnns-with-pytorch-2fbf1a4172a3

# Finds all melids with instrument atribute instrument, then
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

# Create data but bin over time (Not used)
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

# One hot encodes pitch data using a dictionary p to say which column each pitch is column
def one_hot_encode(notes, p):
    # Create a dictionary to map note pitches to indices
    
    # Initialize an empty array for one-hot encoded sequences
    one_hot_sequence = np.zeros((len(notes), len(p)))

    # Fill in the one-hot encoding
    v = np.zeros
    i = 100000
    for j, note in enumerate(notes):
        if note > 80:
            i = p['80']
        elif note < 46:
            i = p['46']
        else:
            if note %2 != 0:
                i = p[str(int(note) + 1)]
            else:
                i = p[str(int(note))]
        one_hot_sequence[j, i] = 1

    return one_hot_sequence

# Saves instruments pitch, duration and step data, finds instruments with find_instrument func
def create_data(instrument = 'p', contour_len = 4, pdf = False):
    data_pitch = []
    data_duration = []
    data_step = []
    data_end = []
    p = {}

    # Get all instrument (default piano) data 
    df, bpms = find_instruments(instrument)
    df = {'0':df[1], '1':df[4]}
    for data_set in df: # iterate through each solo
        step = 0 #Base step size
        #Only one full tones
        p = {'46':0, '48':0, '50':0, '52':0, '54':0, '56':0, '58':0, '60':0, '62':0, '64':0, '66':0, '68':0, '70':0, '72':0, '74':0, '76':0, '78':0,'80':0}
        
        #Semi tones between 54 - 70
        #p = {'46':0, '48':0, '50':0, '52':0, '54':0, '55':0, '56':0, '57':0, '58':0, '59':0, '60':0, '61':0, '62':0, '63':0, '64':0, '65':0, '66':0, '67':0, '68':0, '69':0, '70':0, '72':0, '74':0, '76':0, '78':0,'80':0}
        
        #All semitones
        #p = {'46':0, '47':0, '48':0, '49':0, '50':0, '51':0, '52':0, '53':0, '54':0, '55':0, '56':0, '57':0, '58':0, '59':0, '60':0, '61':0, '62':0, '63':0, '64':0, '65':0, '66':0, '67':0, '68':0, '69':0, '70':0, '71':0, '72':0, '73':0, '74':0, '75':0, '76':0, '77':0, '78':0, '79':0,'80':0}

        p_to_i = np.zeros(len(p.keys()))
        for i, n in enumerate(p):
            p[n] = i

        dp_avg = []
        for i in range(len(df[data_set]['pitch'])): # count for each pitch in the df
            data_duration.append(df[data_set]['duration'].iloc[i]) # add duration of current event
            data_pitch.append(df[data_set]['pitch'].iloc[i]) # add pitch of current event
            dp_avg.append(df[data_set]['pitch'].iloc[i])
            
            data_step.append(step)
            if i < len(df[data_set]['pitch']) - 1:
                step = df[data_set]['onset'].iloc[i + 1] - df[data_set]['onset'].iloc[i] - df[data_set]['duration'].iloc[i] # length of silence after current note

    # Convert data to either pandas df or to torch tensor
    data_pitch = one_hot_encode(data_pitch, p)
    df = np.array(data_pitch)
    if pdf:
        df = pd.DataFrame(df, columns=['pitch','step','duration', 'end'])
    else:
        df = torch.from_numpy(df).float()

    return df

# Creates a sequence of length sequence length for the training loop (Window size)
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
    
# Class with model in it, includes func to init hidden layer and func to forward the training
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
        #print(out.shape)
        # print(out)
        # h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        # out, _ = self.rnn(x, h0)
        out = self.fc(out[-1,:])
        #out = self.fc(out)
        return out, hidden
    
    def init_hidden(self):
        #print(self.num_layers, self.hidden_size)
        h0 = torch.zeros(self.num_layers, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.hidden_size)
        #print(f'h0: {h0.shape}')
        #print(f'c0: {c0.shape}')
        #print(h0)
        return (h0, c0)

# Runs the training loop
def train_model(X, y, model, rate, epochs):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Training loop
    
    for epoch in range(epochs):
        n = 0
        for Xi, yi in zip(X, y):
            Xi = Xi.to(device)
            yi = yi.to(device)
            hidden = model.init_hidden()
            outputs, hidden = model(Xi, hidden)
            loss = loss_func(outputs, yi)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n+= 1
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Total runs: {n}')

# Test the model, used only in testModel.py
def test_model(model, start_sequence, max_length):
    model.eval()
    input_seq = torch.tensor(start_sequence, dtype=torch.float32)
    generated_sequence = start_sequence
    hidden = model.init_hidden()
    
    for _ in range(max_length):
        output, hidden = model(input_seq, hidden)
        predicted_event = output.detach().unsqueeze(0).numpy()
        generated_sequence = np.concat((generated_sequence, predicted_event), 0)
        input_seq = torch.cat((input_seq, torch.tensor(predicted_event)), dim = 0) # add predicted event to next input

        input_seq = input_seq[1:,:]
    
    return generated_sequence   

# After saving the model, call pythonosc functionality
def send_saved_model_notification():
    client_ip = "10.17.244.147"  # Localhost IP address for testing
    client_port = 6070        # Port to send messages to MAX/MSP

    def send_message():
        client = udp_client.SimpleUDPClient(client_ip, client_port)
        client.send_message("/model_saved", "Model training completed and saved.")
        print(f"Notification sent to {client_ip}:{client_port}")

    send_message()

if __name__ == "__main__":
    #Load data
    d = create_data()
    
    # Old Hyperparameter Setup
    #input_size = 18
    #hidden_size = 256
    #output_size = 18
    #rate = 0.001
    #epochs = 3
    #sequence_length = 16
    #layers = 5

    #Error catch and search mode
    if len(argv) != 9:
        print('need 8 args')
        exit()

    #Load arguments
    #to run: python3 dataLoad.py seed sequence_length layers epochs input_size output_size rate hidden_size
    seed = int(argv[1])
    sequence_length = int(argv[2])
    layers = int(argv[3])
    epochs = int(argv[4])
    input_size = int(argv[5])
    output_size = int(argv[6])
    rate = float(argv[7])
    hidden_size = int(argv[8])

    #Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up the model
    X, y = make_sequences(d, sequence_length)
    model = rnn_model(input_size, hidden_size, output_size, layers)

    # Record parameters
    print('-----------------------')
    print('Input Size:', input_size)
    print('Output Size:',output_size)
    print('Hidden Size:',hidden_size)
    print('Learning Rate:',rate)
    print('Layers:',layers)
    print('Epochs:',epochs)
    print('-----------------------')
    
    # Train the model
    train_model(X, y, model, rate, epochs)
    
    # Open a file and use dump() 
    with open('model.pkl', 'wb') as file: 
        pickle.dump(model, file) 

    send_saved_model_notification()
    
