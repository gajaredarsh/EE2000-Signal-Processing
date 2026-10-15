import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import params


def gen_bits(n, seed=params.PRBS_SEED, taps=params.PRBS_TAPS,
             reg_len=params.PRBS_LEN):
    """
    Generate n pseudo-random bits using a Fibonacci LFSR.

    RTL REFERENCE (the Verilog LFSR must match this bit-for-bit):

        reg [6:0] lfsr;
        wire feedback = lfsr[0] ^ lfsr[6];
        wire out_bit  = lfsr[0];                 // read BEFORE shifting
        always @(posedge clk)
            lfsr <= {feedback, lfsr[6:1]};        // shift right, insert
                                                   // feedback at bit 6
        initial lfsr = 7'b1111111;                // seed = 0x7F

    taps=(1,7) in this function's 1-indexed convention corresponds to
    XOR-ing bit 0 and bit 6 (0-indexed) above. reg_len=7 gives PRBS-7,
    period 2^7-1=127, ones-fraction = 64/127 ~= 0.5039.
    """
    if seed == 0:
        raise ValueError("LFSR seed must be non-zero")

    state = seed & ((1 << reg_len) - 1)
    bits = np.zeros(n, dtype=np.uint8)
    t0, t1 = taps

    for i in range(n):
        out_bit = state & 1
        fb = ((state >> (t0 - 1)) ^ (state >> (t1 - 1))) & 1
        state = (state >> 1) | (fb << (reg_len - 1))
        bits[i] = out_bit

    return bits


def _prbs7_rtl_reference(n, seed=0x7F):
    """
    Independent, hardcoded, bit-literal mirror of the Verilog snippet
    in gen_bits()'s docstring, used ONLY as a cross-check that the
    generic taps=(1,7) implementation above really matches the intended
    RTL convention (not just "has the right period").
    """
    lfsr = seed & 0x7F
    bits = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        out_bit = lfsr & 1                      # lfsr[0]
        fb = (lfsr & 1) ^ ((lfsr >> 6) & 1)      # lfsr[0] ^ lfsr[6]
        lfsr = (fb << 6) | (lfsr >> 1)           # shift right, insert at bit6
        bits[i] = out_bit
    return bits

def map_qpsk(bits):
    """
    Map a bitstream to QPSK I/Q symbols using Gray coding.
    See LOCKED table in params.py.
    """
    bits = np.asarray(bits)
    if len(bits) % 2 != 0:
        raise ValueError("bit stream length must be even")

    pairs = bits.reshape(-1, 2)
    b0 = pairs[:, 0].astype(float)
    b1 = pairs[:, 1].astype(float)

    I = 1.0 - 2.0 * b0
    Q = 1.0 - 2.0 * b1
    return I, Q


def demap_qpsk_reference(I, Q):
    """Reference inverse of map_qpsk, for Part A's own self-checks only."""
    b0 = (np.asarray(I) < 0).astype(np.uint8)
    b1 = (np.asarray(Q) < 0).astype(np.uint8)
    bits = np.empty(2 * len(I), dtype=np.uint8)
    bits[0::2] = b0
    bits[1::2] = b1
    return bits

def nco_float(n_samples, fc_over_fs=params.FC_OVER_FS, init_phase=0.0):
    """Ideal floating-point carrier generator."""
    n = np.arange(n_samples)
    phase = 2 * np.pi * fc_over_fs * n + init_phase
    return np.cos(phase), np.sin(phase)

def modulate(I, Q, cos, sin, sps=params.SAMPLES_PER_SYMBOL):
    """Float-domain TX mixer: s[n] = I_up[n]*cos[n] - Q_up[n]*sin[n]."""
    I = np.asarray(I)
    Q = np.asarray(Q)
    n_expected = len(I) * sps

    if len(cos) < n_expected or len(sin) < n_expected:
        raise ValueError("carrier arrays too short for given I/Q length and sps")

    I_up = np.repeat(I, sps)
    Q_up = np.repeat(Q, sps)
    return I_up * cos[:n_expected] - Q_up * sin[:n_expected]
