
from aux_nndet import get_complete_model, custom_loss
from aux_matlab import decode_matlab_file
from pymatbridge import Matlab

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import TerminateOnNaN, ModelCheckpoint

import numpy as np
import tensorflow as tf
import hdf5storage

import os
from keras import backend as K

# GPU allocation
K.clear_session()
tf.reset_default_graph()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
# Tensorflow memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.
K.tensorflow_backend.set_session(tf.Session(config=config))

# System parameters
num_tx, num_rx = 4, 4
mod_size = 4
# Architecture parameters
hidden_dim = [250, 150]
num_iters  = 6
# Training parameters
batch_size    = 2048 # Smaller for smaller training data
num_epochs    = 500
learning_rate = 1e-4
# Which SNR are we training at
train_snr_idx   = -2 # 10, 15 or 20 - these must match the file
if mod_size == 4:
    train_snr_array = [0, 5, 10]
elif mod_size == 6:
    train_snr_array = [10, 15, 20]
elif mod_size == 8:
    train_snr_array = [15, 20, 25]
train_snr_value = train_snr_array[train_snr_idx]

# Load bitmaps
contents = hdf5storage.loadmat('matlab/split_bitmaps_mod%d.mat' % mod_size)
real_bitmap, imag_bitmap = contents['real_bitmap'], contents['imag_bitmap']

# Load training data
if mod_size == 4:
    train_file = 'matlab/data/extended_rayleigh_ml_mimo%dby%d_mod%d_seed1234.mat' % (num_rx, num_tx, mod_size)
if mod_size == 6:
    train_file = 'matlab/data/extended_ml_mimo%dby%d_mod%d_seed1357.mat' % (num_rx, num_tx, mod_size)
elif mod_size == 8:
    train_file = 'matlab/data/extended_rayleigh_zf-sic_mimo%dby%d_mod%d_seed1357.mat' % (num_rx, num_tx, mod_size)
contents      = hdf5storage.loadmat(train_file)
ref_y         = np.squeeze(np.asarray(contents['ref_y']))
ref_h         = np.squeeze(np.asarray(contents['ref_h']))
ref_labels    = np.squeeze(np.asarray(contents['ref_labels']))
constellation = np.squeeze(np.asarray(contents['axis_constellation']))
# Reshapes
y_train = np.moveaxis(ref_y[train_snr_idx], -1, -2)
y_train = np.reshape(y_train, (-1, num_rx))
h_train = np.moveaxis(ref_h[train_snr_idx], -1, -3)
h_train = np.reshape(h_train, (-1, num_rx, num_tx))
labels_train = np.moveaxis(ref_labels[train_snr_idx], -1, -3)
labels_train = np.reshape(labels_train, (-1, 2, num_tx))
# Convert labels to one-hot
labels_train = to_categorical(labels_train-1, num_classes=2**(mod_size//2))

# Construct hy
hy_train = np.matmul(np.conj(np.transpose(h_train, (0, 2, 1))), y_train[..., None])[..., 0]
# Construct x randomly starting point - in the constellation
x_train = 1/np.sqrt(2) * (np.random.normal(size=(y_train.shape[0], num_tx)) +
                   1j * np.random.normal(size=(y_train.shape[0], num_tx)))
# Construct hx
hx_train = np.matmul(
        np.matmul(np.conj(np.transpose(h_train, (0, 2, 1))), h_train),
        x_train[..., None])[..., 0]

# Split into real/imaginary
hy_real_train, hy_imag_train = np.real(hy_train), np.imag(hy_train)
x_real_train, x_imag_train   = np.real(x_train), np.imag(x_train)
hx_real_train, hx_imag_train = np.real(hx_train), np.imag(hx_train)
h_real_train, h_imag_train   = np.real(h_train), np.imag(h_train)

# Split into training/validation
hy_real_train, hy_real_val, hy_imag_train, hy_imag_val, \
x_real_train, x_real_val, x_imag_train, x_imag_val, \
hx_real_train, hx_real_val, hx_imag_train, hx_imag_val, \
h_real_train, h_real_val, h_imag_train, h_imag_val, \
labels_train, labels_val = \
 train_test_split(hy_real_train, hy_imag_train, 
                  x_real_train, x_imag_train,
                  hx_real_train, hx_imag_train,
                  h_real_train, h_imag_train,
                  labels_train,
                  test_size=0.2,
                  random_state=2020)
 
# Result directory
global_dir = 'nndet_models'
if not os.path.exists(global_dir):
    os.makedirs(global_dir)
# Local directory
local_dir = global_dir + '/mod%d_layers%d_dim%d_%d_snr%.1f' % (
        mod_size, num_iters, hidden_dim[0], hidden_dim[1], train_snr_value)
if not os.path.exists(local_dir):
    os.makedirs(local_dir)
    
# Instantiate model
full_model, prob_model = get_complete_model(num_tx, num_rx, mod_size,
                                            constellation, hidden_dim, 
                                            num_iters)

# Optimizer
optimizer = Adam(lr=learning_rate)
# Compile with symbol-wise cross-entropy
prob_model.compile(optimizer=optimizer, loss=custom_loss)

# Best weights
best_weights = ModelCheckpoint(local_dir + '/best_weights.h5',
                               monitor='val_loss', save_best_only=True,
                               save_weights_only=True)

# Train
history = prob_model.fit(x=[x_real_train, x_imag_train, hy_real_train, hy_imag_train,
                            hx_real_train, hx_imag_train, h_real_train, h_imag_train],
                         y=labels_train, epochs=num_epochs, batch_size=batch_size,
                         validation_data=([x_real_val, x_imag_val, hy_real_val, hy_imag_val,
                                           hx_real_val, hx_imag_val, h_real_val, h_imag_val],
 labels_val), callbacks=[best_weights, TerminateOnNaN()])

# Load best weights
prob_model.load_weights(local_dir + '/best_weights.h5')

# Load test data
test_file     = 'matlab/data/extended_rayleigh_ml_mimo%dby%d_mod%d_seed4321.mat' % (num_rx, num_tx, mod_size)
contents      = hdf5storage.loadmat(test_file)
test_y        = np.squeeze(np.asarray(contents['ref_y']))
test_h        = np.squeeze(np.asarray(contents['ref_h']))
test_llr      = np.squeeze(np.asarray(contents['ref_llr']))
ref_bits      = np.asarray(contents['ref_bits'])
num_snr, num_codewords = test_llr.shape[:2]
# Reshapes
y_test = np.moveaxis(test_y, -1, -2)
y_test = np.reshape(y_test, (-1, num_rx))
h_test = np.moveaxis(test_h, -1, -3)
h_test = np.reshape(h_test, (-1, num_rx, num_tx))

# Construct hy
hy_test = np.matmul(np.conj(np.transpose(h_test, (0, 2, 1))), y_test[..., None])[..., 0]
# Construct x randomly starting point - in the constellation
x_test = 1/np.sqrt(2) * (np.random.normal(size=(y_test.shape[0], num_tx)) +
                       1j * np.random.normal(size=(y_test.shape[0], num_tx)))
# Construct hx
hx_test = np.matmul(
        np.matmul(np.conj(np.transpose(h_test, (0, 2, 1))), h_test),
        x_test[..., None])[..., 0]

# Split into real/imaginary
hy_real_test, hy_imag_test = np.real(hy_test), np.imag(hy_test)
x_real_test, x_imag_test   = np.real(x_test), np.imag(x_test)
hx_real_test, hx_imag_test = np.real(hx_test), np.imag(hx_test)
h_real_test, h_imag_test   = np.real(h_test), np.imag(h_test)

# Predict test probabilities
probs_test = prob_model.predict([x_real_test, x_imag_test, hy_real_test, hy_imag_test,
                                 hx_real_test, hx_imag_test, h_real_test, h_imag_test],
batch_size=1024)
# Estimate LLRs for bits from the real part
real_ones  = np.matmul(probs_test[:, 0, :, :], real_bitmap.T[None, ...])
real_zeros = np.matmul(probs_test[:, 0, :, :], 1 - real_bitmap.T[None, ...])
real_llr   = np.log(real_zeros / real_ones)
# Same for imaginary part
imag_ones  = np.matmul(probs_test[:, 1, :, :], imag_bitmap.T[None, ...])
imag_zeros = np.matmul(probs_test[:, 1, :, :], 1 - imag_bitmap.T[None, ...])
imag_llr   = np.log(imag_zeros / imag_ones)

# Concatenate on axis
llr_hat = np.concatenate((real_llr, imag_llr), axis=-1)
# Reshape
llr_hat = np.reshape(llr_hat, (-1, mod_size*num_tx))

# Start Matlab engine
eng = Matlab()
eng.start()
# Move to right path
eng.run_code('cd /path/to/Matlab/')

# Dispatch LDPC decoder
bler_sota, ber_sota, _ = decode_matlab_file(eng, 'ldpc',
                                            llr_hat, ref_bits, 
                                            num_snr, num_codewords,
                                            mode='direct')
# Save
hdf5storage.savemat(local_dir +'/results_test.mat', {'bler_sota': bler_sota,
                                                     'ber_sota': ber_sota,
                                                     'snr_range': contents['snr_range']})
# Close Matlab engine
eng.stop()
# Free up memory
del test_llr, ref_h, h_test, test_h, ref_y, y_test