"""
channel.py
Part B of the QPSK FPGA project golden model: the AWGN CHANNEL.

Two noise paths live here:

  1. FLOAT path  (unchanged API: awgn(), gaussian_noise(), lfsr_gaussian(), ...)
     Used by the BER sweeps in main.py. mode="numpy" is the ideal reference,
     mode="lfsr" is the hardware-style generator in floating point.

  2. INTEGER path (new: awgn_int(), noise_gain(), export_noise_vectors(), ...)
     A bit-exact model of the noise block that will go into Verilog. This is what the
     RTL testbench must be compared against, because every step is integer arithmetic.

Verification lives in test_part_b.py (run with: pytest -v).

Author: [FILL IN YOUR NAME] (Part B)
AI-assistance note: [FILL IN per course AI-usage policy]

CHANGELOG (v2):
  - lfsr_bits() now reads a precomputed one-period table (same output, ~100x faster).
    The original bit-by-bit loop is kept as lfsr_bits_reference() and the tests prove the
    two are identical.
  - Added the integer noise path and six fixed LFSR seeds (NOISE_SEEDS).
  - Added gain table, realized-Eb/N0 check and hex export of noise vectors.
  - check_gaussian() now also returns (mean, var) so it can be asserted.
  - The float path is unchanged, so main.py / rx_frontend.py keep working.

---------------------------------------------------------------------------------------
Hardware noise block: exact arithmetic (this is what the Verilog must do)
---------------------------------------------------------------------------------------
  six 16-bit LFSRs, fixed initial states NOISE_SEEDS (below)
      feedback fb = s[0]^s[2]^s[3]^s[5];  s_next = {fb, s[15:1]};  output bit = s_next[0]
      (the output bit is read AFTER the shift); 8 bits form a byte, FIRST bit is the MSB
  raw   = byte0 + ... + byte5 - 765              signed 11 bit, range -765..+765
  prod  = raw * gain                             signed 11 x unsigned 13 -> 24-bit signed
  noise = (prod + 128) >>> 8                     arithmetic shift, 16-bit signed
  sum   = s13 + noise                            s13 = TX mixer output (13-bit, 2047 = 1.0)
                                                 16-bit signed, cannot overflow
  r12   = saturate12( (sum + 4) >>> 3 )          RX input word, 12-bit, 256 = amplitude 1.0

  gain  = round( sigma_tx / 181.02 * 256 ),  sigma_tx = 2047 * sqrt(4 / (Eb/N0))
  (signal power is fixed at 1.0 for constant-envelope QPSK, so the gain is just a table
   indexed by the Eb/N0 switch setting; see gain_table()).

The >>>3 maps the TX scale (2047 = 1.0) to the RX scale (256 = 1.0). It is applied to signal
and noise together, so it does not change Eb/N0. Known limits of the LFSR noise: the sum of
6 bytes is not exactly Gaussian (excess kurtosis -0.2, tails cut at +/-4.2 sigma) and the
sequence repeats every 65535 samples. Both are harmless after the 16-sample integrate-and-dump
and for demo-length runs, but state them in the report.
"""

import os
from functools import lru_cache

import numpy as np

import params
import fixedpoint
from params import SAMPLES_PER_SYMBOL as SPS
from params import CARRIER_CYCLES_PER_SYMBOL as CYCLES


# =======================================================================================
# 1. Eb/N0 -> noise variance (float domain, unit-power signal)
# =======================================================================================

def noise_sigma2(ebn0_db, sps=SPS, power=1.0):
    # QPSK: Es = power*sps, Eb = Es/2, sigma2 = power*sps / (4*Eb/N0)
    ebn0 = 10**(ebn0_db / 10)
    return power * sps / (4 * ebn0)


# =======================================================================================
# 2. 16-bit LFSR
# =======================================================================================

LFSR_PERIOD = 65535


def _lfsr_step(state):
    fb = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
    return (state >> 1) | (fb << 15)


def lfsr_bits_reference(n, seed):
    """Bit-literal reference (slow). RTL convention: fb = s[0]^s[2]^s[3]^s[5];
    s_next = {fb, s[15:1]}; output bit = s_next[0] (read AFTER the shift)."""
    state = seed & 0xFFFF
    if state == 0:
        state = 1
    bits = []
    for i in range(n):
        bit = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
        state = (state >> 1) | (bit << 15)
        bits.append(state & 1)
    return bits


@lru_cache(maxsize=1)
def _lfsr_tables():
    """One full period: position of every state in the cycle, and the output bit per step."""
    states = np.empty(LFSR_PERIOD, dtype=np.int64)
    s = 1
    for k in range(LFSR_PERIOD):
        states[k] = s
        s = _lfsr_step(s)
    assert s == 1, "LFSR is not maximal length"
    pos = np.zeros(1 << 16, dtype=np.int64)
    pos[states] = np.arange(LFSR_PERIOD)
    out_bits = (states[(np.arange(LFSR_PERIOD) + 1) % LFSR_PERIOD] & 1).astype(np.uint8)
    return states, pos, out_bits


def lfsr_bit_array(n, seed):
    """Same bits as lfsr_bits_reference(), as a uint8 numpy array, read from the period table."""
    _, pos, out_bits = _lfsr_tables()
    state = seed & 0xFFFF
    if state == 0:
        state = 1
    idx = (pos[state] + np.arange(n)) % LFSR_PERIOD
    return out_bits[idx]


def lfsr_bits(n, seed):
    """16 bit LFSR bit stream as a list (kept for compatibility)."""
    return lfsr_bit_array(n, seed).tolist()


def lfsr_byte_stream(n_bytes, seed):
    """8 bits make a byte, first bit is the MSB."""
    bits = lfsr_bit_array(n_bytes * 8, seed).reshape(n_bytes, 8).astype(np.int64)
    weights = 2**np.arange(7, -1, -1)
    return bits.dot(weights)


# =======================================================================================
# 3. FLOAT noise path (unchanged behaviour)
# =======================================================================================

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
    mean, var = np.mean(x), np.var(x)
    print(label, "mean =", mean, " var =", var)
    return mean, var


def q_func(x):
    from math import erfc, sqrt
    return 0.5 * erfc(x / sqrt(2))


def theory_ber(ebn0_db):
    return q_func(np.sqrt(2 * 10**(ebn0_db / 10)))


# =======================================================================================
# 4. INTEGER noise path (bit-exact model of the Verilog noise block)
# =======================================================================================

NOISE_K = 6                                           # bytes summed per noise sample
NOISE_MEAN = 765                                      # 6 * 127.5, exactly an integer
NOISE_RAW_STD = (NOISE_K * (256**2 - 1) / 12) ** 0.5  # 181.02
NOISE_GAIN_SHIFT = 8                                  # gain is a fixed-point number with 8 fraction bits
NOISE_GAIN_BITS = 13                                  # unsigned gain register width
TX_TO_RX_SHIFT = 3                                    # 13-bit TX scale (2047=1.0) -> 12-bit RX scale (256=1.0)
RX_BITS = getattr(params, "RX_IN_BITS", 12)           # RX input word width (same default as Part C)

# Six FIXED initial LFSR states, spaced 10922 steps apart along the 65535-state cycle
# (starting at step 4000 so none of them is a sparse state). The RTL uses the same values.
NOISE_SEEDS = getattr(params, "NOISE_SEEDS",
                      (0x5CDF, 0x660A, 0x9884, 0xD5B9, 0x4276, 0x3359))

# Eb/N0 settings selectable on the board (index = switch value)
EBN0_TABLE_DB = tuple(range(0, 16))                   # 0 .. 15 dB in 1 dB steps


def lfsr_noise_raw(n, seeds=NOISE_SEEDS):
    """Integer noise: sum of six byte streams minus 765. Range -765..+765 (11-bit signed)."""
    total = np.zeros(n, dtype=np.int64)
    for s in seeds:
        total += lfsr_byte_stream(n, s)
    return total - NOISE_MEAN


def noise_gain(ebn0_db, shift=NOISE_GAIN_SHIFT, full_scale=params.DATAPATH_FULL_SCALE):
    """Unsigned gain register value for a requested Eb/N0 (signal power fixed at 1.0)."""
    sigma_tx = np.sqrt(noise_sigma2(ebn0_db, SPS, 1.0)) * full_scale
    g = int(round(sigma_tx / NOISE_RAW_STD * (1 << shift)))
    if not 0 <= g < (1 << NOISE_GAIN_BITS):
        raise ValueError(f"Eb/N0 = {ebn0_db} dB needs gain {g}, which does not fit "
                         f"{NOISE_GAIN_BITS} unsigned bits")
    return g


def realized_ebn0_db(gain, shift=NOISE_GAIN_SHIFT, full_scale=params.DATAPATH_FULL_SCALE):
    """Eb/N0 that a given integer gain actually produces (gain is rounded, so not exact)."""
    sigma_unit = gain * NOISE_RAW_STD / (1 << shift) / full_scale
    return 10 * np.log10(SPS / (4 * sigma_unit**2))


def gain_table(ebn0_list=EBN0_TABLE_DB, shift=NOISE_GAIN_SHIFT):
    """Gain register values for each switch setting (this table becomes a ROM in Verilog)."""
    return [noise_gain(e, shift) for e in ebn0_list]


def noise_path_widths(max_gain=(1 << NOISE_GAIN_BITS) - 1, shift=NOISE_GAIN_SHIFT):
    """Worst-case signed width of each stage of the hardware noise path."""
    def signed_bits(m):                       # bits needed for range [-m, +m]
        return int(np.ceil(np.log2(m + 1))) + 1
    raw_max = NOISE_MEAN                      # +/-765
    prod_max = raw_max * max_gain
    noise_max = (prod_max + (1 << (shift - 1))) >> shift
    mixer_max = int(np.ceil(params.DATAPATH_FULL_SCALE * np.sqrt(2)))   # worst-case TX mixer output
    sum_max = noise_max + mixer_max
    return {"raw": signed_bits(raw_max), "prod": signed_bits(prod_max),
            "noise": signed_bits(noise_max), "sum": signed_bits(sum_max)}


def noise_int(n, gain, seeds=NOISE_SEEDS, shift=NOISE_GAIN_SHIFT):
    """Scaled integer noise in the TX scale: (raw*gain + 2^(shift-1)) >> shift."""
    raw = lfsr_noise_raw(n, seeds)
    return (raw * int(gain) + (1 << (shift - 1))) >> shift


def awgn_int(s13, ebn0_db=None, gain=None, seeds=NOISE_SEEDS, shift=NOISE_GAIN_SHIFT,
             return_wide=False):
    """
    Bit-exact channel. Input: integer TX mixer output s13 (13-bit, 2047 = amplitude 1.0).
    Output: integer 12-bit RX words (256 = amplitude 1.0), saturated.
    Give either ebn0_db (gain looked up) or the integer gain directly.
    return_wide=True also returns the 16-bit sum, for debugging the RTL adder.
    """
    s13 = np.asarray(s13)
    if not np.issubdtype(s13.dtype, np.integer):
        raise TypeError("awgn_int() takes integer TX samples (use tx_nco.modulate_fixed)")
    s13 = s13.astype(np.int64)
    if gain is None:
        if ebn0_db is None:
            raise ValueError("give ebn0_db or gain")
        gain = noise_gain(ebn0_db, shift)
    total = s13 + noise_int(len(s13), gain, seeds, shift)
    r12 = fixedpoint.saturate((total + (1 << (TX_TO_RX_SHIFT - 1))) >> TX_TO_RX_SHIFT, RX_BITS)
    return (r12, total) if return_wide else r12


def export_noise_vectors(out_dir, n_samples=2048, ebn0_db=6.0):
    """Hex files for the Verilog noise-block testbench (one word per line, $readmemh)."""
    os.makedirs(out_dir, exist_ok=True)
    g = noise_gain(ebn0_db)
    raw = lfsr_noise_raw(n_samples)
    scaled = noise_int(n_samples, g)
    fixedpoint.write_hex(os.path.join(out_dir, "noise_raw.hex"), raw, 11)
    fixedpoint.write_hex(os.path.join(out_dir, "noise_scaled.hex"), scaled, 16)
    fixedpoint.write_hex(os.path.join(out_dir, "noise_gain_table.hex"), gain_table(), NOISE_GAIN_BITS)
    fixedpoint.write_hex(os.path.join(out_dir, "noise_seeds.hex"), list(NOISE_SEEDS), 16)
    with open(os.path.join(out_dir, "noise_info.txt"), "w") as f:
        f.write(f"samples exported    : {n_samples}\n")
        f.write(f"noise_raw.hex       : 11-bit signed, six-byte sum minus {NOISE_MEAN}\n")
        f.write(f"noise_scaled.hex    : 16-bit signed, (raw*gain + {1 << (NOISE_GAIN_SHIFT - 1)}) >> {NOISE_GAIN_SHIFT}\n")
        f.write(f"gain used           : {g} (Eb/N0 = {ebn0_db} dB, realized {realized_ebn0_db(g):.3f} dB)\n")
        f.write(f"noise_gain_table.hex: {len(EBN0_TABLE_DB)} entries, {NOISE_GAIN_BITS}-bit UNSIGNED, "
                f"Eb/N0 = {EBN0_TABLE_DB[0]}..{EBN0_TABLE_DB[-1]} dB\n")
        f.write("noise_seeds.hex     : six 16-bit LFSR initial states\n")
    return g


if __name__ == "__main__":
    print("Gain table (switch setting -> gain register -> realized Eb/N0):")
    for e in EBN0_TABLE_DB:
        g = noise_gain(e)
        print(f"  {e:2d} dB -> gain {g:5d} -> {realized_ebn0_db(g):6.3f} dB")
    print("\nWorst-case signed widths:", noise_path_widths())
    print("\nRun `pytest -v` to verify this file (test_part_b.py).")
