import sqlite3
import pandas as pd
import numpy as np

con = sqlite3.connect('assets/wjazzd.db')

solo_info = pd.read_sql_query("SELECT * FROM solo_info", con)
melody = pd.read_sql_query("SELECT * FROM melody", con)

def find_instruments(ins): # Finds melids of solos of a certain instrument
    melid = []
    for i in range(456):
        if solo_info['instrument'][i] == str(ins):
            melid += [solo_info['melid'][i]]

    return melid

def split_data(ins): # Returns dict of dataframes
    melid = find_instruments(ins)
    data = {}

    for i, m in enumerate(melid):
        data[i] = melody[melody['melid'] == m]

    return data

def create_data(bin_size, instrument = 'p', contour_len = 4):
    data_pitch = []
    data_contour = []

    df = split_data('p')

    for data_set in data:
        #initial data key
        k = df[data_set]['pitch'].keys()[0]
        k0 = k
        kf = df[data_set]['pitch'].keys()[-1]
        t = 0
        dur = df[data_set]['duration'][k]
        tot_dur = 0

        data_pitch_temp = []
        data_contour_temp = []
        
        while k < kf + 1:
            dur = df[data_set]['duration'][k]
            data_pitch += [df[data_set]['pitch'][k]]
            if k - k0 > (contour_len - 1):
                data_contour += [np.average(df[data_set]['pitch'][(k - k0 - contour_len):(k - k0)], 
                    weights=df[data_set]['duration'][(k - k0 - contour_len):(k - k0)])]
            else:
                data_contour += [0]

            t += bin_size

            if t >  dur:
                k += 1
                t = 0

        data_pitch.extend(data_pitch_temp)
        data_contour.extend(data_contour_temp)


    d = np.array([data_pitch, data_contour]).T
    new_df = pd.DataFrame(d, columns=['pitch','contour'])

    return new_df