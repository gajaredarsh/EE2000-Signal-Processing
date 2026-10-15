"""
fixedpoint.py
Part B of the QPSK FPGA project golden model: shared quantizer and hex helpers.

Used by channel.py (Part B), rx_frontend.py (Part C) and rx_backend_eval.py (Part D).
Verification lives in test_part_b.py (run with: pytest -v).

Author: [FILL IN YOUR NAME] (Part B)
AI-assistance note: [FILL IN per course AI-usage policy]

CHANGELOG (v2):
  - Added saturate(x, bits): integer saturation, the same rule a hardware saturator uses.
    (rx_frontend.py has a private _sat() that does the same thing; they can share this one.)
  - Added read_hex(): inverse of write_hex(), used to round-trip test the exported files.
  - Documented the signed range used by to_int().
  - __main__ no longer writes test.hex into the repo folder.

Range convention (important for the Verilog saturators)
--------------------------------------------------------
Every signed b-bit word here uses the FULL two's-complement range [-2^(b-1), 2^(b-1) - 1],
so for 12 bits that is [-2048, 2047]. This is slightly different from the TX NCO, whose
outputs stay symmetric at +/-2047 (-2048 never occurs). That is fine: the RTL saturator for
a b-bit word must clip to exactly [-2^(b-1), 2^(b-1) - 1] to match to_int() and saturate().
"""

import numpy as np


def quantize(x, bits, full_scale=1.0):
    """Round to a signed `bits`-bit grid and return FLOAT values (same units as x)."""
    levels = 2**(bits - 1)
    codes = np.round(np.array(x) * levels / full_scale)
    codes = np.clip(codes, -levels, levels - 1)
    return codes * full_scale / levels


def to_int(x, bits, full_scale=1.0):
    """Float -> signed integer codes: round(x * 2^(bits-1) / full_scale), clipped to the full range."""
    levels = 2**(bits - 1)
    codes = np.round(np.array(x) * levels / full_scale)
    codes = np.clip(codes, -levels, levels - 1)
    return codes.astype(int)


def saturate(x, bits):
    """Clip an INTEGER array to the signed `bits`-bit range [-2^(bits-1), 2^(bits-1)-1]."""
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError("saturate() is for integer arrays; use to_int() to quantize floats")
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return np.clip(x, lo, hi)


def clip_fraction(x, bits, full_scale=1.0):
    """Fraction of samples that would be clipped by to_int(x, bits, full_scale)."""
    levels = 2**(bits - 1)
    codes = np.round(np.array(x) * levels / full_scale)
    clipped = (codes < -levels) | (codes > levels - 1)
    return np.mean(clipped)


def to_hex(codes, bits):
    """Signed integer codes -> list of two's-complement hex strings, ceil(bits/4) digits each."""
    width = (bits + 3) // 4
    mask = (1 << bits) - 1
    out = []
    for c in codes:
        out.append(format(int(c) & mask, '0%dx' % width))
    return out


def write_hex(filename, codes, bits):
    """Write one hex word per line, ready for Verilog $readmemh."""
    lines = to_hex(codes, bits)
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def read_hex(filename, bits):
    """Read a file written by write_hex() back into signed integers."""
    sign = 1 << (bits - 1)
    full = 1 << bits
    vals = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16)
            vals.append(v - full if v >= sign else v)
    return np.array(vals, dtype=np.int64)


if __name__ == "__main__":
    print("fixedpoint.py: helper module. Run `pytest -v` to verify it (test_part_b.py).")
