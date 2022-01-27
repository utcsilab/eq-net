clc, clear, close all

channel_type = 'fading';
code_type = 'ldpc';
num_packets = 10000;
num_tx = 2;
num_rx = 2;
num_data_per_channel = 6; % How many data symbols re-use the same channel
mod_size = 6;
% snr_range = linspace(16, 24, 9); % For 2x2 64-QAM ML
snr_range   = linspace(11.5, 15.5, 10);
% snr_range = linspace(1, 10, 10); % For 4x4 16-QAM ML
% snr_range = linspace(10, 22, 1);
% snr_range = [10 12.5 15];
bit_seed = 4567;
target_folder = 'data_tiled';
% Which decoding algorithm
algorithm  = 'ml';

[bler, ber, time_elapsed] = GenMIMOTrainingData(channel_type, code_type, ...
    algorithm, alg_params, num_packets, ...
    num_tx, num_rx, num_data_per_channel, ...
    mod_size, snr_range, bit_seed, target_folder)

% % Generate validation data in the same shot
% num_packets = 10000;
% bit_seed = 4321;
% 
% [bler, ber] = GenMIMOTrainingData(channel_type, code_type, algorithm, num_packets, ...
%     num_tx, num_rx, mod_size, snr_range, bit_seed, target_folder)
% 
% % [bler, ber] = GenTrainingData(channel_type, code_type, num_packets, ...
% %             mod_size, snr_range, bit_seed, target_folder)
