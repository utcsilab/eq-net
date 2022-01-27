% Inputs: Code type, Channel type, Number of packets, SNR Range, Seed
% Code type = {'ldpc', 'polar'}
% Channel type = {'rayleigh', 'fading'}
% Outputs: Reference BLER/BER
% Byproducts: Writes reference bit files and LLR files in 'target_folder'
function [bler, ber, time_elapsed] = GenMIMOTrainingData(channel_type, code_type, algorithm, ...
    alg_params, num_packets, num_tx, num_rx, num_data_per_channel, ...
    mod_size, snr_range, bit_seed, target_folder)

% Modulation order
mod_order = 2^mod_size;
% Interleaver seed - do NOT generally change
inter_seed = 1111;

% Code parameters
if strcmp(code_type, 'ldpc')
    % Taken from IEEE 802.11n: HT LDPC matrix definitions
    % You can change this according to your needs
    Z = 27;
    rotmatrix = ...
        [0 -1 -1 -1 0 0 -1 -1 0 -1 -1 0 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
        22 0 -1 -1 17 -1 0 0 12 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1;
        6 -1 0 -1 10 -1 -1 -1 24 -1 0 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1;
        2 -1 -1 0 20 -1 -1 -1 25 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1;
        23 -1 -1 -1 3 -1 -1 -1 0 -1 9 11 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1;
        24 -1 23 1 17 -1 3 -1 10 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1;
        25 -1 -1 -1 8 -1 -1 -1 7 18 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1;
        13 24 -1 -1 0 -1 8 -1 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1;
        7 20 -1 16 22 10 -1 -1 23 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1;
        11 -1 -1 -1 19 -1 -1 -1 13 -1 3 17 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1;
        25 -1 8 -1 23 18 -1 14 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0;
        3 -1 -1 -1 16 -1 -1 2 25 5 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0];

    H = zeros(size(rotmatrix)*Z);
    Zh = diag(ones(1,Z),0);

    % Convert into binary matrix
    for r=1:size(rotmatrix,1)
        for c=1:size(rotmatrix,2)
            rotidx = rotmatrix(r,c);
            if (rotidx > -1)
                Zt = circshift(Zh,[0 rotidx]);
            else
                Zt = zeros(Z);
            end
            limR = (r-1)*Z+1:r*Z;
            limC = (c-1)*Z+1:c*Z;
            H(limR, limC) = Zt;
        end
    end

    hEnc = comm.LDPCEncoder('ParityCheckMatrix', sparse(H));
    hDec = comm.LDPCDecoder('ParityCheckMatrix', sparse(H), 'DecisionMethod', 'Soft decision', ...
        'IterationTerminationCondition', 'Parity check satisfied', 'MaximumIterationCount', 50);
    % System parameters
    K = size(H, 1);
    N = size(H, 2);
elseif strcmp(code_type, 'polar')
    % Polar code parameters
    K = 128;
    N = 256;
    L = 4; % List length
else
    error('Invalid code type!')
end

% Channel type
if strcmp(channel_type, 'rayleigh')
    % Passthrough
elseif strcmp(channel_type, 'fading')
    % Universal dirac impulse
    dirac_in = zeros(N, num_tx);
else
    error('Invalid channel type!')
end

% Compute number of MIMO transmission required to fill one codeword
% We call this "batch_size"
if mod(N, mod_size * num_tx) > 0
    padded_llrs = N - floor(N / (mod_size * num_tx)) * (mod_size * num_tx);
else
    padded_llrs = 0;
end
batch_size = floor(N / (mod_size * num_tx));

% Interleaver
rng(inter_seed);
P = randperm(N);
R(P) = 1:N;
rng(bit_seed)

% Auxiliary tables for fast LLR computation
bitmap = de2bi(0:(mod_order-1)).';
constellation = qammod(bitmap, mod_order, 'InputType', 'bit', 'UnitAveragePower', true);
% Store
filename = sprintf('constellation%d.mat', mod_size);
save(filename, 'bitmap', 'constellation', '-v7.3');
% Get constellation on axis
axis_constellation = unique(real(constellation));

% Construct a real-axis bitmap
real_bitmap = -1 * ones(mod_size/2, 2^(mod_size/2));
imag_bitmap = -1 * ones(mod_size/2, 2^(mod_size/2));

% For each real value
for value_idx = 1:numel(axis_constellation)
    % Find all constellation symbols with that real part
    real_symbols = constellation(real(constellation) == axis_constellation(value_idx));
    % Find indices of those symbols
    [~, idx] = max(constellation == real_symbols.', [], 2);
    % Get bitmap there from first half only
    local_bitmap = bitmap(1:mod_size/2, idx);
    
    % Sum across the second axis
    bitmap_sum = sum(local_bitmap, 2);
    % All zeroes means bit is always zero there
    % Any value between 0-2^(mod_size/2) is inconclusive
    bitmap_sum((bitmap_sum > 0) & (bitmap_sum < 2^(mod_size/2))) = -1;
    % All ones means bit is always one there
    bitmap_sum(bitmap_sum == 2^(mod_size/2)) = 1;
    
    % Find all locations where sum is zero/max - bit is all zero/one    
    % Mark them down
    real_bitmap(bitmap_sum == 0, value_idx) = 0;
    real_bitmap(bitmap_sum == 1, value_idx) = 1;
    
    % Repeat for imaginary part
    imag_symbols = constellation(imag(constellation) == axis_constellation(value_idx));
    % Find indices of those symbols
    [~, idx] = max(constellation == imag_symbols.', [], 2);
    % Get bitmap there from second half only
    local_bitmap = bitmap(mod_size/2+1:end, idx);
    
    % Sum across the second axis
    bitmap_sum = sum(local_bitmap, 2);
    % All zeroes means bit is always zero there
    % Any value between 0-2^(mod_size/2) is inconclusive
    bitmap_sum((bitmap_sum > 0) & (bitmap_sum < 2^(mod_size/2))) = -1;
    % All ones means bit is always one there
    bitmap_sum(bitmap_sum == 2^(mod_size/2)) = 1;
    
    % Find all locations where sum is zero/max - bit is all zero/one    
    % Mark them down
    imag_bitmap(bitmap_sum == 0, value_idx) = 0;
    imag_bitmap(bitmap_sum == 1, value_idx) = 1;
end
% Save to file
save(sprintf('split_bitmaps_mod%d.mat', mod_size), 'real_bitmap', 'imag_bitmap')

% Performance metrics
ber  = zeros(numel(snr_range), num_packets);
bler = zeros(numel(snr_range), num_packets);

% Byproducts
ref_bits = zeros(numel(snr_range), num_packets, K);
ref_llr  = zeros(numel(snr_range), num_packets, num_tx, mod_size, batch_size);
% More byproducts
ref_y      = zeros(numel(snr_range), num_packets, num_rx, batch_size);
ref_h      = zeros(numel(snr_range), num_packets, num_rx, num_tx, batch_size);
ref_x      = zeros(numel(snr_range), num_packets, num_tx, batch_size);
ref_n      = zeros(numel(snr_range), num_packets);
ref_labels = zeros(numel(snr_range), num_packets, 2, num_tx, batch_size);
% Progress
progressbar(0, 0);
% Profiling
time_elapsed = 0;

% Main loop
for snr_idx = 1:numel(snr_range)
    noise_power = 10 ^ (-snr_range(snr_idx)/10);
    for run_idx = 1:num_packets
        % Random bits
        payload_bits = randi([0 1], K, 1);
        % Save in collection
        ref_bits(snr_idx, run_idx, :) = payload_bits;
        
        % Encode bits
        if strcmp(code_type, 'ldpc')
            bitsEnc = hEnc(payload_bits);
        elseif strcmp(code_type, 'polar')
            bitsEnc = nrPolarEncode(payload_bits, N);
        end
        
        % Interleave bits
        bitsInt = bitsEnc(P);
        
        % Modulate bits
        x = qammod(bitsInt(1:end-padded_llrs), mod_order, 'InputType', 'bit', 'UnitAveragePower', true);
        % Reshape to batched MIMO size
        x_mimo = reshape(x, [num_tx, 1, batch_size]);
        
        % Fill labels
        for symbol_idx = 1:numel(axis_constellation)
            % Add labels for real and imaginary parts
            ref_labels(snr_idx, run_idx, 1, :, :) = squeeze(ref_labels(snr_idx, run_idx, 1, :, :)) + ...
                (real(squeeze(x_mimo)) == axis_constellation(symbol_idx)) * symbol_idx;
            ref_labels(snr_idx, run_idx, 2, :, :) = squeeze(ref_labels(snr_idx, run_idx, 2, :, :)) + ...
                (imag(squeeze(x_mimo)) == axis_constellation(symbol_idx)) * symbol_idx;
        end
        
        % Channel effects
        % AWGN
        n = 1/sqrt(2) * sqrt(noise_power) * (randn(num_rx, 1, batch_size) + 1i * randn(num_rx, 1, batch_size));
        if strcmp(channel_type, 'rayleigh')
            % Draw random N(0, 1)
            h = 1/sqrt(2) * (randn(num_rx, num_tx, batch_size) + 1i * randn(num_rx, num_tx, batch_size));
            % Check that we can split and repeat
            assert(mod(batch_size, num_data_per_channel) == 0)
            % Downselect
            h = h(:, :, 1:num_data_per_channel:end);
            % Tile
            h = repelem(h, 1, 1, num_data_per_channel);
        elseif strcmp(channel_type, 'fading')
            % Output channel
            hFreq = zeros(N / (mod_size*num_tx), num_rx, num_tx);
            
            % Fading channel
            cdl = nrCDLChannel;
            % Reset seed every packet
            cdl.Seed = inter_seed+snr_idx+run_idx;
            cdl.DelayProfile = 'CDL-A';
            cdl.DelaySpread = 316e-9; % Street canyon @ 6 GHz
            cdl.CarrierFrequency = 6e9;
            cdl.MaximumDopplerShift = 0;
            % Configure MIMO, single antenna on panel
            cdl.TransmitAntennaArray.Size = [1 1 1 1 num_tx];
            cdl.ReceiveAntennaArray.Size  = [1 1 1 1 num_rx];
            % Larger antenna/panel spacing for improved conditioning
            cdl.TransmitAntennaArray.ElementSpacing = [1 1 2 2];
            cdl.ReceiveAntennaArray.ElementSpacing = [1 1 2 2];
            
            % Derive impulse response using antenna selection
            for antenna_idx = 1:num_tx
                % Empty vector
                dirac_in = zeros(N / (mod_size*num_tx), num_tx);
                % Antenna selection
                dirac_in(1, antenna_idx) = 1;
                
                % Get delay domain signal
                hImp = cdl(dirac_in);
                % Convert to frequency and store
                hFreq(:, :, antenna_idx) = fft(hImp);
                
            end
            
            % Downsample channel
            h = hFreq;
            % Permute dimensions
            h = permute(h, [2, 3, 1]);
            
            % Normalize to unit average power
            h = h ./ sqrt(mean(abs(h) .^ 2, 'all'));
        end
        
        % For some algorithms, perform column-sort here, without loss of
        % generality
        if strcmp(algorithm, 'm-algorithm')
            rx_power = squeeze(sum(abs(h) .^ 2, 1));
            [~, sorted_idx] = sort(rx_power, 1, 'ascend');
            % Replace each channel with its column sorted version
            for channel_idx = 1:size(h, 3)
                local_h  = h(:, :, channel_idx);
                sorted_h = local_h(:, sorted_idx(:, channel_idx));
                h(:, :, channel_idx) = sorted_h;
            end
        end
        % Apply channel
        y = tmult(h, x_mimo) + n;
        
        if ~strcmp(algorithm, 'none')
            % Profile
            start_time = tic;
            % LLR computation
            llrInt = ComputeLLRMIMO(constellation, bitmap, mod_size, y, h, noise_power, algorithm, ...
                alg_params, x_mimo, n);
            % Profile
            time_elapsed = time_elapsed + toc(start_time);

            % Save in collection
            ref_llr(snr_idx, run_idx, :, :, :) = llrInt;
        end
        % Save auxiliaries as well
        ref_x(snr_idx, run_idx, :, :)    = squeeze(x_mimo);
        ref_y(snr_idx, run_idx, :, :)    = squeeze(y);
        ref_h(snr_idx, run_idx, :, :, :) = h;
        ref_n(snr_idx, run_idx)          = noise_power;
        
        % Terminate early if no algorithm is used
        if strcmp(algorithm, 'none')
            continue
        end
        
        % Flatten correctly
        llrInt = permute(llrInt, [2, 1, 3]);
        llrInt = llrInt(:);
        % Pad with erasures
        llrInt = [llrInt; zeros(padded_llrs, 1)]; %#ok<AGROW>
        % Deinterleave bits
        llrDeint = double(llrInt(R));
        
        % Channel decoder
        if strcmp(code_type, 'ldpc')
            if strcmp(algorithm, 'zf-sic')
                % Limit maximum LLR value
                llrDeint(abs(llrDeint) >= 6) = 6 * sign(llrDeint(abs(llrDeint) >= 6));
            end
            llrOut = hDec(llrDeint);
            bitsEst = (sign(-llrOut) +1) / 2;
        elseif strcmp(code_type, 'polar')
            bitsEst = nrPolarDecode(llrDeint, K, N, L);
        end
        
        % Determine bit/packet error
        ber(snr_idx, run_idx)  = mean(payload_bits ~= bitsEst);
        bler(snr_idx, run_idx) = ~all(payload_bits == bitsEst);

        % Progress
        progressbar(double(run_idx) / double(num_packets), []);
    end
    % Progress
    progressbar([], snr_idx / numel(snr_range));
end
progressbar(1, 1)

% Average results
ber  = mean(ber, 2);
bler = mean(bler, 2);

% Save byproducts
filename = sprintf('%s/extended_%s_%s_mimo%dby%d_mod%d_seed%d.mat', ...
    target_folder, channel_type, algorithm, num_tx, num_rx, mod_size, bit_seed);
save(filename, 'ref_bits', 'ref_llr', 'ref_x', 'ref_y', 'ref_h', 'ref_n', 'snr_range', ...
    'num_data_per_channel', 'bler', 'ber', 'ref_labels', 'axis_constellation', '-v7.3');

end