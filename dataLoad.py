import sqlite3
import pandas as pd

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

data = split_data('p')

print(data[1].keys())
print(data[1][['duration','beatdur']])

past = 1
time = 0
beat_time = 0
its = 0
for i in data[1]['bar'].keys():
    
    if data[1]['bar'][i] == past:
        time += data[1]['duration'][i]
        beat_time += data[1]['beatdur'][i]
        its += 1
    else:
        print(past, time, beat_time, its)
        past = data[1]['bar'][i]
        time = data[1]['duration'][i]
        beat_time = data[1]['beatdur'][i]
        its = 0
