from midiutil.MidiFile import MIDIFile
from midi2audio import FluidSynth


def makeMidi(d):
    # create your MIDI object
    mf = MIDIFile(1)     # only 1 track
    track = 0   # the only track

    time = 0    # start at the beginning
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, 120)

    # add some notes
    channel = 0
    volume = 100

    for i in d:
        print(i)
        mf.addNote(track, channel, int(i[0]), time, i[2], volume)
        time += i[2]

    # write it to disk
    with open("output.mid", 'wb') as outf:
        mf.writeFile(outf)


## add an output number of 128 so that the data knows when to stop
