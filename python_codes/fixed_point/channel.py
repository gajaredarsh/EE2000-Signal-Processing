import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from params import SAMPLES_PER_SYMBOL as SPS
from params import CARRIER_CYCLES_PER_SYMBOL as CYCLES

def noise_sigma2(ebn0_db, sps=SPS, power=1.0):
    # QPSK: Es = power*sps, Eb = Es/2, sigma2 = power*sps / (4*Eb/N0)
    ebn0 = 10**(ebn0_db / 10)
    return power * sps / (4 * ebn0)

def lfsr_bits(n, seed):
    # 16 bit LFSR, used to build pseudo random noise bytes
    # RTL convention: fb = s[0]^s[2]^s[3]^s[5];  s_next = {fb, s[15:1]};
    # output bit = s_next[0] (read AFTER the shift); 8 bits make a byte, first bit is the MSB
    state = seed & 0xFFFF
    if state == 0:
        state = 1
    bits = []
    for i in range(n):
        bit = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
        state = (state >> 1) | (bit << 15)
        bits.append(state & 1)
    return bits

def lfsr_byte_stream(n_bytes, seed):
    bits = lfsr_bits(n_bytes * 8, seed)
    bits = np.array(bits).reshape(n_bytes, 8)
    weights = 2**np.arange(7, -1, -1)
    return bits.dot(weights)

def lfsr_gaussian(n, seed=1, k=6):
    # sum of k uniform bytes approximates a gaussian (central limit theorem)
    total = np.zeros(n)
    for i in range(k):
        stream_seed = (seed * 1009 + i * 97) % 65535 + 1   # 1..65535, never 0
        total += lfsr_byte_stream(n, stream_seed)
    mean = k * 127.5
    var = k * (256**2 - 1) / 12
    return (total - mean) / np.sqrt(var)

def gaussian_noise(n, mode="numpy", seed=0):
    if mode == "numpy":
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n)
    elif mode == "lfsr":
        return lfsr_gaussian(n, seed=seed if seed else 1)
    else:
        raise ValueError("mode must be numpy or lfsr")

def awgn(s, ebn0_db, mode="numpy", seed=0, sps=SPS):
    s = np.array(s, dtype=float)
    power = np.mean(s**2)          # measured, so float and fixed point signals both work
    sigma = np.sqrt(noise_sigma2(ebn0_db, sps, power))
    noise = gaussian_noise(len(s), mode, seed)
    return s + sigma * noise

def check_gaussian(x, label=""):
    print(label, "mean =", np.mean(x), " var =", np.var(x))

def q_func(x):
    from math import erfc, sqrt
    return 0.5 * erfc(x / sqrt(2))

def theory_ber(ebn0_db):
    return q_func(np.sqrt(2 * 10**(ebn0_db/10)))


