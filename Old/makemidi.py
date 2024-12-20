from midiutil.MidiFile import MIDIFile
from midi2audio import FluidSynth
from plotnine import *
import pandas

def makeMidi(d):
    # create your MIDI object
    mf = MIDIFile(deinterleave=False)     # only 1 track
    track = 0   # the only track
    time = 0    # start at the beginning
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, 60)

    # add some notes
    channel = 1
    volume = 50
    # data = {}
    # times = []
    # pitches = []
    file_index = 0
    done = False
    for i in d:
        pitch = 0
        
        try:
            step = i[1]
            duration = i[2]

            pitch = int(i[0])
        except:
            pitch = int(i)
            step = 0
            duration = 1
        end = 0
        
        if end == "dfakfb": # end of solo
            # print(f'adding note: {track, channel, pitch, time, duration, volume}')
            mf.addNote(track, channel, pitch, time, duration, volume)
            with open(f"output_{file_index}.mid", 'wb') as outf:
                mf.writeFile(outf)
            print(f"created output_{file_index}.mid")
            file_index += 1
            time = 0
            mf = MIDIFile(1) # initialize new midi file
            mf.addTrackName(track, time, "Sample Track")
            mf.addTempo(track, time, 60)
            done = True
        else:
            # print(f'adding note: {track, channel, pitch, time, duration, volume}')
            mf.addNote(track, channel, pitch, time, duration, volume)
            time += duration + step


    # write it to disk
    with open("output.mid", 'wb') as outf:
        mf.writeFile(outf)