"""
params.py
Shared system parameters for the QPSK FPGA project golden model.

Author: [FILL IN - TEAM AGREED FILE]

CHANGELOG:
  v1: initial defaults.
  v2 (review round 1):
    - PRBS_TAPS corrected (7,6) -> (1,7). See gen_bits() docstring.
    - Gray-coding comment table corrected to match code's actual formula.
  v3 (review round 2):
    - Added MIXER_OUT_BITS = 13. The TX mixer computes I*cos - Q*sin
      with I,Q in {+1,-1} and cos,sin quantized to DATAPATH_BITS=12
      (full_scale = 2047). Worst case |I*cos - Q*sin| occurs at the
      +/-45 deg LUT points: 2047*sqrt(2) ~= 2895, which does NOT fit
      in 12-bit signed (max 2047) but DOES fit in 13-bit signed
      (max 4095). See required_mixer_bits() in tx_nco.py, which derives
      this number from DATAPATH_BITS so it stays correct if DATAPATH_BITS
      ever changes.
    - Added Es/Eb documentation for Part B's AWGN scaling (see below).
    - Documented the exact PRBS bit-level convention for the Verilog
      LFSR (see gen_bits() in tx_nco.py for the matching RTL snippet).

Es / Eb reference for Part B (channel.py):
    In the ideal (float, unscaled) domain, symbols I,Q in {+1,-1} and
    the carrier has unit amplitude. Averaged over a full symbol period
    (which contains an integer number of carrier cycles, since
    CARRIER_CYCLES_PER_SYMBOL=2), the per-sample signal power is:
        E[s[n]^2] = I^2/2 + Q^2/2 = 1.0
    With SAMPLES_PER_SYMBOL=16 samples per symbol:
        Es = power * sps = 1.0 * 16 = 16   (energy per symbol, sample units)
        Eb = Es / 2 = 8                    (QPSK carries 2 bits/symbol)
    This holds for the UNSCALED float domain used by nco_float()/modulate().
    It does NOT directly apply to the fixed-point integer outputs of
    nco_fixed()/modulate_fixed(), which are scaled by DATAPATH_FULL_SCALE
    and must be renormalized before computing Eb/N0-based noise power.
"""

import numpy as np

# ---------------------------------------------------------------
# Top-level timing parameters
# ---------------------------------------------------------------
SAMPLES_PER_SYMBOL = 16          # NCO/sample-clock ticks per QPSK symbol
CARRIER_CYCLES_PER_SYMBOL = 2    # carrier completes 2 full cycles per symbol

FC_OVER_FS = CARRIER_CYCLES_PER_SYMBOL / SAMPLES_PER_SYMBOL   # = 0.125

# ---------------------------------------------------------------
# NCO (carrier generator) parameters
# ---------------------------------------------------------------
NCO_ACC_BITS = 16
NCO_LUT_BITS = 8
NCO_LUT_SIZE = 1 << NCO_LUT_BITS
NCO_PHASE_INC = round(FC_OVER_FS * (1 << NCO_ACC_BITS))   # = 8192

# Deliberately "generic" ratio used ONLY by the precision-sweep
# self-tests in tx_nco.py (never used in the real signal chain).
GENERIC_TEST_FC_OVER_FS = 0.12398

# ---------------------------------------------------------------
# Fixed-point datapath
# ---------------------------------------------------------------
DATAPATH_BITS = 12
DATAPATH_FULL_SCALE = (1 << (DATAPATH_BITS - 1)) - 1   # = 2047

# Mixer output width. MUST be >= required_mixer_bits(DATAPATH_BITS)
# (checked by an assertion in tx_nco.py's self-test). Derivation:
# worst case |I*cos - Q*sin| = DATAPATH_FULL_SCALE * sqrt(2) ~= 2895,
# which needs 13-bit signed (max 4095), not 12-bit (max 2047).
MIXER_OUT_BITS = 13

# ---------------------------------------------------------------
# PRBS bit source
# ---------------------------------------------------------------
PRBS_LEN = 7
PRBS_TAPS = (1, 7)           # see gen_bits() docstring for RTL mapping
PRBS_SEED = 0x7F

# ---------------------------------------------------------------
# Gray-coded QPSK mapping (LOCKED -- Part D's demapper must invert
# this exactly)
#
#   Rule implemented in map_qpsk(): I = 1 - 2*b0,  Q = 1 - 2*b1
#
#   (b0,b1)=(0,0) -> (I,Q)=(+1,+1)   45 deg
#   (b0,b1)=(0,1) -> (I,Q)=(+1,-1)  315 deg
#   (b0,b1)=(1,1) -> (I,Q)=(-1,-1)  225 deg
#   (b0,b1)=(1,0) -> (I,Q)=(-1,+1)  135 deg
#
#   Gray check (walking by angle): 45(00)->135(10)->225(11)->315(01)->45(00)
#   each step flips exactly one bit.
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# TX mixer sign convention (LOCKED):  s[n] = I*cos[n] - Q*sin[n]
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Receiver constants (v4). These used to be defined at the top of rx_frontend.py; they live
# here now so floating_point/ and fixed_point/ always use the SAME numbers.
# ---------------------------------------------------------------
FIR_ORDER = 24                 # order = taps - 1  (24 -> 25 taps)
FIR_FC = 0.125                 # -6 dB cutoff, cycles per sample
FIR_WINDOW = "hamming"
COEF_BITS = 12                 # signed FIR coefficients, format Q1.(COEF_BITS-1)
RX_IN_BITS = 12                # receiver input word (the "ADC" word)
RX_FULL_SCALE = 8.0            # float amplitude that maps to full scale at the receiver input
PASS_EDGE = 1.0 / SAMPLES_PER_SYMBOL                         # data band edge (cycles/sample)
STOP_EDGE = 2.0 * FC_OVER_FS - 1.0 / SAMPLES_PER_SYMBOL      # start of the 2*fc junk lobe