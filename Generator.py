import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


class Generator:

    def __init__(self, mod='weights.hdf5'):
        self.model = mod

    def consolidate_notes(self, a, b):
        la = len(a)
        lb = len(b)
        final_notes = []
        if lb < la:
            sl = [a[i] for i in range(len(b)) if a[i] not in b]
            final_notes = b
            for i in range(len(sl)):
                final_notes.append(sl[i])
                if len(final_notes) == len(a):
                    break
        else:
            final_notes = a[0:len(b)]
        return final_notes

    def generate(self):
        with open('data/notes', 'rb') as filepath:
            pre_notes = pickle.load(filepath)

        post_notes = self.get_notes()

        notes=self.consolidate_notes(pre_notes, post_notes)

        pitchnames = sorted(set(item for item in notes))
        n_vocab = len(set(notes))

        network_input, normalized_input = self.prepare_sequences(notes, pitchnames, n_vocab)
        cat_net_in, net_out = self.prepare_sequences_categorical(notes, n_vocab)
        model = self.create_network(normalized_input, n_vocab)
        self.train(model, cat_net_in, net_out, 10)
        prediction_output = self.generate_notes(model, network_input, pitchnames, n_vocab)
        self.create_midi(prediction_output)

    def prepare_sequences(self, notes, pitchnames, n_vocab):
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        sequence_length = 100
        network_input = []
        output = []
        for i in range(0, len(notes) - sequence_length):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)
        normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        normalized_input = normalized_input / float(n_vocab)

        return (network_input, normalized_input)

    def prepare_sequences_categorical(self, notes, n_vocab):
        """ Prepare the sequences used by the Neural Network """
        sequence_length = 100

        # get all pitch names
        pitchnames = sorted(set(item for item in notes))

        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

        network_input = []
        network_output = []

        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)

        network_output = np_utils.to_categorical(network_output)

        return (network_input, network_output)

    def get_notes(self):
        """ Get all the notes and chords from the midi files in the ./midi_songs directory """
        notes = []

        for file in glob.glob("midi_songs/*.mid"):
            midi = converter.parse(file)

            print("Parsing %s" % file)

            notes_to_parse = None

            try: # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse() 
            except: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        with open('data/new_notes', 'wb') as filepath:
            pickle.dump(notes, filepath)

        return notes

    def create_network(self, network_input, n_vocab):
        """ create the structure of the neural network """
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            recurrent_dropout=0.3,
            return_sequences=True
        ))
        model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
        model.add(LSTM(512))
        model.add(BatchNorm())
        model.add(Dropout(0.3))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(BatchNorm())
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.load_weights(self.model)

        return model

    def train(self,model, network_input, network_output, eps=200):
        """ train the neural network """
        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]

        model.fit(network_input, network_output, epochs=eps, batch_size=128, callbacks=callbacks_list)

    def generate_notes(self, model, network_input, pitchnames, n_vocab):
        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

        pattern = network_input[start]
        prediction_output = []
        for note_index in range(500):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

        return prediction_output

    def create_midi(self, prediction_output):
        offset = 0
        output_notes = []
        for pattern in prediction_output:
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Violin()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            offset += 0.5

        midi_stream = stream.Stream(output_notes)

        midi_stream.write('midi', fp='test_output.mid')




if __name__ == '__main__':
    g = Generator()
    g.generate()