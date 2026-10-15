import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import signal
import params
import fixedpoint
from fixed_point import tx_nco

SPS = params.SAMPLES_PER_SYMBOL
FC_OVER_FS = params.FC_OVER_FS
FIR_ORDER = params.FIR_ORDER
FIR_FC = params.FIR_FC
FIR_WINDOW = params.FIR_WINDOW
COEF_BITS = params.COEF_BITS
RX_IN_BITS = params.RX_IN_BITS
RX_FULL_SCALE = params.RX_FULL_SCALE
PASS_EDGE = params.PASS_EDGE
STOP_EDGE = params.STOP_EDGE

def _sat(x, bits):
    """Saturate an integer array to a signed `bits`-wide range (what a hardware saturator does)."""
    hi = (1 << (bits - 1)) - 1
    lo = -(1 << (bits - 1))
    return np.clip(x, lo, hi)


def group_delay_samples(h):
    """Delay of a symmetric odd-length FIR, in samples: (taps-1)/2.  Raises if not linear-phase."""
    h = np.asarray(h)
    if len(h) % 2 == 0 or not np.allclose(h, h[::-1]):
        raise ValueError("filter is not a symmetric odd-length (linear-phase type I) FIR")
    return (len(h) - 1) // 2


def _fir(x, h, aligned):
    """
    y[n] = sum_k h[k] * x[n-k]   (exact for int64 arrays).
    aligned=False : causal, zero start-up state, output delayed by D samples.
    aligned=True  : D zeros are flushed through and the first D outputs dropped, so y[n]
                    lines up with x[n].  Same length as x either way.
    """
    x = np.asarray(x)
    if aligned:
        d = group_delay_samples(h)
        xp = np.concatenate([x, np.zeros(d, dtype=x.dtype)])
        return np.convolve(xp, h)[d:d + len(x)]
    return np.convolve(x, h)[:len(x)]


def quantize_rx(r, bits=RX_IN_BITS, full_scale=RX_FULL_SCALE):
    """
    Turn the float received signal into the integer word the FPGA receiver would see.
    Uses Part B's shared quantizer: code = round(r * 2^(bits-1) / full_scale), clipped.
    Default: 12 bits, full scale 8.0, so 1.0 (the TX amplitude) = 256 codes.
    Why full scale 8: at Eb/N0 = 0 dB the noise sigma is 2.0, so +/-8 is +/-4 sigma.
    """
    return fixedpoint.to_int(r, bits, full_scale)


def downconvert_fixed(r_int, cos_int, sin_int, lo_bits=params.DATAPATH_BITS, out_bits=RX_IN_BITS):
    """
    Integer RX mixer, the exact model of what rx_mixer.v will compute.
        xi = ( r_int *  cos_int + 2^(lo_bits-2) ) >>> (lo_bits-1)
        xq = ( r_int * -sin_int + 2^(lo_bits-2) ) >>> (lo_bits-1)
    cos_int / sin_int come from Part A's nco_fixed() (full scale 2047).  Shifting by 11
    (divide by 2048) brings the product back to the same scale as r_int, so the output
    needs no more bits than the input.  The +2^10 before the shift is round-to-nearest.
    """
    r_int = np.asarray(r_int, dtype=np.int64)
    c = np.asarray(cos_int[:len(r_int)], dtype=np.int64)
    s = np.asarray(sin_int[:len(r_int)], dtype=np.int64)
    shift = lo_bits - 1
    rnd = 1 << (shift - 1)
    xi = (r_int * c + rnd) >> shift
    xq = (-(r_int * s) + rnd) >> shift
    return _sat(xi, out_bits), _sat(xq, out_bits)

def quantize_coeffs(h, bits=COEF_BITS, preserve_dc=True):
    """
    Float taps -> signed integers in Q1.(bits-1):  h_int = round(h * 2^(bits-1)).
    Uses Part B's to_int().  With preserve_dc=True the rounding leftover is pushed into the
    centre tap so that sum(h_int) = 2^(bits-1) EXACTLY (DC gain of precisely 1.0) while the
    filter stays symmetric (the centre tap has no mirror partner).
    """
    h_int = fixedpoint.to_int(h, bits, full_scale=1.0)
    if preserve_dc:
        h_int[len(h_int) // 2] += (1 << (bits - 1)) - int(h_int.sum())
    return h_int


def coeffs_to_float(h_int, bits=COEF_BITS):
    """Back to float so the quantized filter can be analysed with the same tools."""
    return np.asarray(h_int, dtype=float) / (1 << (bits - 1))


def acc_bits_needed(h_int, x_bits=RX_IN_BITS):
    """
    Width of the accumulator that can never overflow (worst case: every sample at full scale
    with the sign that matches each tap).  Handy when writing the Verilog.
    """
    worst = int(np.abs(h_int).sum()) * (1 << (x_bits - 1))
    return int(np.ceil(np.log2(worst + 1))) + 1      # +1 for the sign bit


def lpf_fixed(x_int, h_int, coef_bits=COEF_BITS, out_bits=RX_IN_BITS, aligned=False):
    """
    Integer FIR, bit-exact model of rx_fir.v:
        acc = sum h_int[k] * x_int[n-k]          (exact, wide accumulator)
        y   = saturate( (acc + 2^(coef_bits-2)) >>> (coef_bits-1) )
    The shift undoes the Q1.(coef_bits-1) scaling of the taps; the added term rounds to nearest.
    """
    acc = _fir(np.asarray(x_int, dtype=np.int64), np.asarray(h_int, dtype=np.int64), aligned)
    shift = coef_bits - 1
    y = (acc + (1 << (shift - 1))) >> shift
    return _sat(y, out_bits)


def rx_frontend_fixed(r_int, cos_int, sin_int, h_int, aligned=False):
    """Integer mixer + integer FIR on both branches.  Returns (yi_int, yq_int)."""
    xi, xq = downconvert_fixed(r_int, cos_int, sin_int)
    return lpf_fixed(xi, h_int, aligned=aligned), lpf_fixed(xq, h_int, aligned=aligned)

def export_rx_vectors(out_dir, h_int, r_int, cos_int, sin_int, n_vec=2048):
    """
    Write $readmemh-style hex files (Part B's write_hex, two's complement, one value per line):
        fir_coeffs.hex            the taps
        rx_r.hex                  mixer/FIR input  (RX_IN_BITS)
        lo_cos.hex, lo_sin.hex    local carrier    (DATAPATH_BITS)
        rx_xi.hex, rx_xq.hex      expected mixer output
        rx_yi.hex, rx_yq.hex      expected FIR output (causal, zero start-up state)
    The Verilog testbench feeds rx_r/lo_* and must reproduce the other files bit for bit.
    """
    os.makedirs(out_dir, exist_ok=True)
    r_v = r_int[:n_vec]
    c_v = cos_int[:n_vec]
    s_v = sin_int[:n_vec]
    xi, xq = downconvert_fixed(r_v, c_v, s_v)
    yi = lpf_fixed(xi, h_int)
    yq = lpf_fixed(xq, h_int)

    def w(name, codes, bits):
        fixedpoint.write_hex(os.path.join(out_dir, name), codes, bits)

    w("fir_coeffs.hex", h_int, COEF_BITS)
    w("rx_r.hex", r_v, RX_IN_BITS)
    w("lo_cos.hex", c_v, params.DATAPATH_BITS)
    w("lo_sin.hex", s_v, params.DATAPATH_BITS)
    w("rx_xi.hex", xi, RX_IN_BITS)
    w("rx_xq.hex", xq, RX_IN_BITS)
    w("rx_yi.hex", yi, RX_IN_BITS)
    w("rx_yq.hex", yq, RX_IN_BITS)
    return xi, xq, yi, yq


