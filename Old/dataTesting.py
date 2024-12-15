import sqlite3
import pandas as pd

con = sqlite3.connect('assets/wjazzd.db')
melody = pd.read_sql_query("SELECT * FROM melody", con)
comp_info = pd.read_sql_query("SELECT * FROM composition_info", con)
track_info = pd.read_sql_query("SELECT * FROM transcription_info", con)
solo_info = pd.read_sql_query("SELECT * FROM solo_info", con)


#Track id: Song
#Melid: solo
c = 0
print(solo_info.keys())
for i in range(456):
    if solo_info['instrument'][i] == 'p':
        print(solo_info['melid'][i], solo_info['rhythmfeel'][i], solo_info['style'][i])

print(solo_info.keys())


#for i in range(800):
#    print(melody['onset'][i])