# -*- coding: utf-8 -*-
# requirement:python3
# Author: binglel
# Last modified: 20181114 11:00
# Email: chyb3.14@gmail.com
# Filename: main
# Description:
# ******************************************************
import time
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from utils import maybe_download as maybe_download
from utils import sparse_tuple_from as sparse_tuple_from

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
# 0 is reserved to space
FIRST_INDEX = ord('a') - 1
print (FIRST_INDEX)

# Some configs
num_features = 13
# Number of units in the LSTM cell
num_units=50
num_classes =26 + 1 + 1
num_epochs = 120
num_hidden = 50
num_layers = 1
batch_size = 1
initial_learning_rate = 1e-2
momentum = 0.9
num_examples = 1
num_batches_per_epoch = int(num_examples/batch_size)
audio_filename = maybe_download('LDC93S1.wav', 93638)
target_filename = maybe_download('LDC93S1.txt', 62)
# fs is framerate
fs, audio = wav.read(audio_filename)
inputs = mfcc(audio, samplerate=fs)
print (inputs.shape)

# Tranform in 3D array
train_inputs = np.asarray(inputs[np.newaxis, :])
print (train_inputs.shape)

# Normalize
train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
print (train_inputs.shape)

train_seq_len = [train_inputs.shape[1]]
with open(target_filename, 'r') as f:
    #Only the last line is necessary
    line = f.readlines()[-1]
    # Get only the words between [a-z] and replace period for none
    original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
    targets = original.replace(' ', '  ')
    targets = targets.split(' ')
print (targets)

targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])
print (targets)

# Transform char into index
targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])
print (targets)

# Creating sparse representation to feed the placeholder
train_targets = sparse_tuple_from([targets])
print (train_targets)

# We don't have a validation dataset :(
val_inputs, val_targets, val_seq_len = train_inputs, train_targets, train_seq_len
graph = tf.Graph()
with graph.as_default():
    '''
    inputs是输入的placeholder
    输入的尺寸是[batch_size, max_stepsize, num_features],
    but the batch_size and max_stepsize can vary along each step
    '''
    inputs = tf.placeholder(tf.float32, [None, None, num_features])
    # Here we use sparse_placeholder that will generate a SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)
    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])
with graph.as_default():
    '''
    Defining the cell
    Can be:
        tf.nn.rnn_cell.RNNCell
        tf.nn.rnn_cell.GRUCell
    '''
    cells = []
    for _ in range(num_layers):
        cell = tf.contrib.rnn.LSTMCell(num_units)  # Or LSTMCell(num_units)
        cells.append(cell)
    stack = tf.contrib.rnn.MultiRNNCell(cells)
    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
    print (outputs)

    shape = tf.shape(inputs)
    print (shape)

    batch_s, max_timesteps = shape[0], shape[1]
    print (batch_s)
    print (max_timesteps)

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])
    print (outputs)

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))
    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b
    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])
    # Time major
    logits = tf.transpose(logits, (1, 0, 2))
    # Option 2:
    # (it's slower but you'll get better results)
print(logits.get_shape())

with graph.as_default():
    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)
# decoder style
with graph.as_default():
    # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

with graph.as_default():
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
session=tf.Session(graph=graph)
with graph.as_default():
   init=tf.global_variables_initializer()
session.run(init)
for curr_epoch in range(num_epochs):
    train_cost = train_ler = 0
    start = time.time()
    for batch in range(num_batches_per_epoch):
        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
        batch_cost, _ = session.run([cost, optimizer], feed)
        train_cost += batch_cost*batch_size
        train_ler += session.run(ler, feed_dict=feed)*batch_size
    train_cost /= num_examples
    train_ler /= num_examples
    val_feed = {inputs: val_inputs, targets: val_targets, seq_len: val_seq_len}
    val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)
    log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
    print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, val_cost, val_ler, time.time() - start))

feed = {inputs: train_inputs,targets: train_targets,seq_len: train_seq_len}
d = session.run(decoded[0], feed_dict=val_feed)
print(d)

str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
print(str_decoded)

# Replacing blank label to none
str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
print(str_decoded)

# Replacing space label to space
str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
print('Original:\n%s' % original)
print('Decoded:\n%s' % str_decoded)

session.close()

