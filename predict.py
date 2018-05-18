import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


SEQ_LENGTH = 100

def buildmodel(VOCABULARY):
    model = Sequential()
    model.add(LSTM(256, input_shape = (SEQ_LENGTH, 1), return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(VOCABULARY, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    return model


file = open('wonderland.txt', encoding = 'utf8')
raw_text = file.read()    #you need to read further characters as well
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
print(chars)

bad_chars = ['#', '*', '@', '_', '\ufeff']
for i in range(len(bad_chars)):
    raw_text = raw_text.replace(bad_chars[i],"")

chars = sorted(list(set(raw_text)))
print(chars)

text_length = len(raw_text)
char_length = len(chars)
VOCABULARY = char_length
print("Text length = " + str(text_length))
print("No. of characters = " + str(char_length))

char_to_int = dict((c, i) for i, c in enumerate(chars))
input_strings = []
output_strings = []

for i in range(len(raw_text) - SEQ_LENGTH):
    X_text = raw_text[i: i + SEQ_LENGTH]
    X = [char_to_int[char] for char in X_text]
    input_strings.append(X)
    Y = raw_text[i + SEQ_LENGTH]
    output_strings.append(char_to_int[Y])

length = len(input_strings)
input_strings = np.array(input_strings)
input_strings = np.reshape(input_strings, (input_strings.shape[0], input_strings.shape[1], 1))
input_strings = input_strings/float(VOCABULARY)

output_strings = np.array(output_strings)
output_strings = np_utils.to_categorical(output_strings)
print(input_strings.shape)
print(output_strings.shape)


model = buildmodel(VOCABULARY)
filepath="saved_models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(input_strings, output_strings, epochs = 50, batch_size = 128, callbacks = callbacks_list)
