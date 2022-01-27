from aux_oampnet2 import get_complete_tensor_model

from sklearn.model_selection import train_test_split
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
# Set global seed
np.random.seed(2020)
# Tensorflow memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.
K.tensorflow_backend.set_session(tf.Session(config=config))

# System parameters
num_tx, num_rx = 4, 4
mod_size = 4
# Architecture parameters
num_iterations = 4
# Training parameters
batch_size    = 100
num_epochs    = 10
learning_rate = 1e-4

# Load bitmaps
contents = hdf5storage.loadmat('constellation%d.mat' % mod_size)
constellation = contents['constellation'] # !!! Has to be swapped for 64-QAM

# Load training data
train_file    = 'matlab/data/extended_rayleigh_ml_mimo%dby%d_mod%d_seed1234.mat' % (num_rx, num_tx, mod_size)
contents      = hdf5storage.loadmat(train_file)
ref_x         = np.squeeze(np.asarray(contents['ref_x']))
ref_y         = np.squeeze(np.asarray(contents['ref_y']))
ref_h         = np.squeeze(np.asarray(contents['ref_h']))
ref_labels    = np.squeeze(np.asarray(contents['ref_labels']))
train_snr_array = np.squeeze(np.asarray(contents['snr_range']))
# Load test data
# test_file       = 'matlab/data/extended_rayleigh_zf-sic_mimo%dby%d_mod%d_seed9999.mat' % (num_rx, num_tx, mod_size)
test_file       = 'matlab/data/extended_rayleigh_ml_mimo%dby%d_mod%d_seed4321.mat' % (num_rx, num_tx, mod_size)
contents        = hdf5storage.loadmat(test_file)
ref_x_test      = np.squeeze(np.asarray(contents['ref_x']))
ref_y_test      = np.squeeze(np.asarray(contents['ref_y']))
ref_h_test      = np.squeeze(np.asarray(contents['ref_h']))
ref_labels_test = np.squeeze(np.asarray(contents['ref_labels']))
test_snr_array  = np.squeeze(np.asarray(contents['snr_range']))
# For each SNR point
for train_snr_idx, train_snr_value in enumerate(train_snr_array):
    # Clear session
    K.clear_session()
    
    # Get noise power
    sigma_n = 10 ** (-train_snr_value / 10)
    # Reshapes
    x_train = np.moveaxis(ref_x[train_snr_idx], -1, -2)
    x_train = np.reshape(x_train, (-1, num_tx))
    y_train = np.moveaxis(ref_y[train_snr_idx], -1, -2)
    y_train = np.reshape(y_train, (-1, num_rx))
    h_train = np.moveaxis(ref_h[train_snr_idx], -1, -3)
    h_train = np.reshape(h_train, (-1, num_rx, num_tx))
    
    # Construct input-x starting at zeroes
    x_input_train = np.zeros((y_train.shape[0], num_tx))
    # Construct v starting with zero estimate
    v_train = (np.square(np.linalg.norm(y_train, axis=-1, keepdims=True)) - num_rx * sigma_n) / np.trace(np.matmul(
            np.conj(np.transpose(h_train, axes=(0, 2, 1))), h_train), axis1=-1, axis2=-2)[..., None]
    v_train = np.real(v_train)
    v_train = np.maximum(v_train, 5e-13)
    # Construct tau starting at ones
    tau_train = np.ones((y_train.shape[0], 1))
    # Split into real/imaginary
    x_input_real_train, x_input_imag_train = np.real(x_input_train), np.imag(x_input_train)
    x_real_train, x_imag_train   = np.real(x_train), np.imag(x_train)
    y_real_train, y_imag_train   = np.real(y_train), np.imag(y_train)
    h_real_train, h_imag_train   = np.real(h_train), np.imag(h_train)
    
    # Split into training/validation
    x_input_real_train, x_input_real_val, x_input_imag_train, x_input_imag_val, \
    x_real_train, x_real_val, x_imag_train, x_imag_val, \
    y_real_train, y_real_val, y_imag_train, y_imag_val, \
    h_real_train, h_real_val, h_imag_train, h_imag_val, \
    v_train, v_val, tau_train, tau_val = \
     train_test_split(x_input_real_train, x_input_imag_train,
                      x_real_train, x_imag_train,
                      y_real_train, y_imag_train,
                      h_real_train, h_imag_train,
                      v_train, tau_train,
                      test_size=0.2,
                      random_state=2020)
     
    # Result directory
    global_dir = 'oampnet2_models'
    if not os.path.exists(global_dir):
        os.makedirs(global_dir)
    # Local directory
    local_dir = global_dir + '/mod%d_layers%d_lr%.4f_batch%d_snr%.2f' % (
            mod_size, num_iterations, learning_rate, batch_size, train_snr_value)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        
    # Instantiate model
    full_model = get_complete_tensor_model(num_tx, num_rx, mod_size,
                                           constellation, sigma_n, 
                                           num_iterations)
    
    # Optimizer
    optimizer = Adam(lr=learning_rate)
    # Compile with symbol-wise cross-entropy
    full_model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Best weights
    best_weights = ModelCheckpoint(local_dir + '/best_weights.h5',
                                   monitor='val_loss', save_best_only=True,
                                   save_weights_only=True)
    
    # Train
    history = full_model.fit(x=[x_input_real_train, x_input_imag_train,
                                y_real_train, y_imag_train,
                                h_real_train, h_imag_train,
                                v_train, tau_train],
                             y=[x_real_train, x_imag_train], epochs=num_epochs, batch_size=batch_size,
                             validation_data=([x_input_real_val, x_input_imag_val,
                                               y_real_val, y_imag_val, 
                                               h_real_val, h_imag_val,
                                               v_val, tau_val],
     [x_real_val, x_imag_val]), callbacks=[best_weights, TerminateOnNaN()])
    
    # Load best weights
    full_model.load_weights(local_dir + '/best_weights.h5')
    
    # Prepare test data
    x_test = np.moveaxis(ref_x_test[train_snr_idx], -1, -2)
    x_test = np.reshape(x_test, (-1, num_tx))
    y_test = np.moveaxis(ref_y_test[train_snr_idx], -1, -2)
    y_test = np.reshape(y_test, (-1, num_rx))
    h_test = np.moveaxis(ref_h_test[train_snr_idx], -1, -3)
    h_test = np.reshape(h_test, (-1, num_rx, num_tx))
    
    # Construct input-x starting at zeroes
    x_input_test = np.zeros((y_test.shape[0], num_tx))
    # Construct v starting with zero estimate
    v_test = (np.square(np.linalg.norm(y_test, axis=-1, keepdims=True)) - num_rx * sigma_n) / np.trace(np.matmul(
            np.conj(np.transpose(h_test, axes=(0, 2, 1))), h_test), axis1=-1, axis2=-2)[..., None]
    v_test = np.real(v_test)
    v_test = np.maximum(v_test, 5e-13)
    # Construct tau starting at ones
    tau_test = np.ones((y_test.shape[0], 1))
    # Split into real/imaginary
    x_input_real_test, x_input_imag_test = np.real(x_input_test), np.imag(x_input_test)
    x_real_test, x_imag_test   = np.real(x_test), np.imag(x_test)
    y_real_test, y_imag_test   = np.real(y_test), np.imag(y_test)
    h_real_test, h_imag_test   = np.real(h_test), np.imag(h_test)
    
    # Get estimated symbols
    x_hat_real_test, x_hat_imag_test = full_model.predict([x_input_real_test, x_input_imag_test,
                                            y_real_test, y_imag_test,
                                            h_real_test, h_imag_test,
                                            v_test, tau_test], batch_size=1024)
    