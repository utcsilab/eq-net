# eq-net
Joint High-Dimensional Soft Bit Estimation and Quantization using Deep Learning

# Instructions
0. Install requirements.txt
1. Generate ML soft bit data using matlab/main.m
2. Train a EQ-Net model using python/train_eqnet.py (after setting the proper 'train_file' and 'test_file')
(optionally, change the architecture, the default architecture is EQ-Net-L with a low latency)
3. (Optional) Train deep baselines using python/train_oampnet2.py and python/train_nndet.py