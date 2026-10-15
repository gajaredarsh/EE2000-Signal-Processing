import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import signal
from scipy.optimize import brentq
from scipy.special import erfc
import matplotlib.pyplot as plt

import params
from floating_point import channel
from floating_point import tx_nco

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

def downconvert(r, cos, sin):
    """
    Float RX mixer.  xi = r*cos,  xq = -r*sin.

    Why it works:  r = I*cos - Q*sin  (+ noise).
        r*cos  = I*cos^2 - Q*sin*cos = I/2 + (I/2)cos(2wt) - (Q/2)sin(2wt)
        -r*sin = -I*cos*sin + Q*sin^2 = Q/2 - (Q/2)cos(2wt) - (I/2)sin(2wt)
    Each output = (wanted data at 0 Hz, level +/-0.5) + (junk at 2*fc).
    The FIR low-pass that follows removes the junk.
    """
    r = np.asarray(r)
    n = len(r)
    if len(cos) < n or len(sin) < n:
        raise ValueError("local carrier arrays are shorter than the received signal")
    return r * cos[:n], -r * sin[:n]


def design_lpf(order=FIR_ORDER, fc=FIR_FC, window=FIR_WINDOW):
    """
    Windowed-sinc linear-phase FIR low-pass.

        order  : number of taps - 1.  MUST be even, so the delay order/2 is a whole number
                 of samples (a "type I" filter).  25 taps -> order 24.
        fc     : -6 dB cutoff in CYCLES PER SAMPLE (0 < fc < 0.5).  0.125 = fs/8.
        window : 'hamming' | 'hann' | 'blackman' | 'boxcar' | ('kaiser', beta)

    Idea: the ideal low-pass has an infinitely long sinc impulse response.  We cut it to
    order+1 taps and smooth the cut with a window so the stopband does not ring.
    Returns float taps h[0..order], symmetric, with sum(h) = 1 (DC gain exactly 1).
    """
    if order % 2 != 0:
        raise ValueError("order must be even (odd order = half-sample delay, awkward for timing)")
    if not (0.0 < fc < 0.5):
        raise ValueError("fc is in cycles/sample and must be between 0 and 0.5")
    return signal.firwin(order + 1, fc, window=window, fs=1.0)


def group_delay_samples(h):
    """Delay of a symmetric odd-length FIR, in samples: (taps-1)/2.  Raises if not linear-phase."""
    h = np.asarray(h)
    if len(h) % 2 == 0 or not np.allclose(h, h[::-1]):
        raise ValueError("filter is not a symmetric odd-length (linear-phase type I) FIR")
    return (len(h) - 1) // 2

def freq_response(h, n_points=4096):
    """Frequency (cycles/sample, 0..0.5) and complex response H(f) of the FIR."""
    return signal.freqz(h, worN=n_points, fs=1.0)


def check_mask(h):
    """
    Measure the filter against the mask above.  Returns a dict (all gains in dB):
        pass_max, pass_min : largest / smallest gain for f <= 0.0625 (data band)
        stop_max           : largest gain for f >= 0.1875 (the 2*fc junk lobe and beyond)
        delay              : group delay in samples
    """
    f, H = freq_response(h, 8192)
    mag_db = 20 * np.log10(np.abs(H) + 1e-300)
    return {
        "pass_max": float(mag_db[f <= PASS_EDGE].max()),
        "pass_min": float(mag_db[f <= PASS_EDGE].min()),
        "stop_max": float(mag_db[f >= STOP_EDGE].max()),
        "delay": group_delay_samples(h),
    }


def finite_zeros(h, tol=1e-9):
    """
    Zeros of H(z). The two outermost taps of a windowed sinc can be exactly 0 (a sinc zero
    crossing), which would put zeros at infinity. We drop such end taps first; those
    "missing" zeros sit at infinity and are returned as a count.
    Returns (finite_zeros, n_at_infinity). The poles of any FIR are all at z = 0.
    """
    h = np.asarray(h, dtype=float)
    keep = np.abs(h) > tol * np.abs(h).max()
    lo, hi = np.argmax(keep), len(h) - np.argmax(keep[::-1])
    z = np.roots(h[lo:hi])
    return z, (len(h) - 1) - len(z)

def _fir(x, h, aligned):
    """
    y[n] = sum_k h[k] * x[n-k]   (the multiply-accumulate chain / shift register in hardware)
    Works for float arrays and for int64 arrays (integer arithmetic is then exact).
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


def lpf(x, h, aligned=False):
    """Float FIR low-pass.  See _fir() for the meaning of `aligned`."""
    return _fir(x, h, aligned)


def rx_frontend(r, cos, sin, h, aligned=False):
    """Mixer + low-pass on both branches (float).  Returns (yi, yq), baseband level +/-0.5."""
    xi, xq = downconvert(r, cos, sin)
    return lpf(xi, h, aligned), lpf(xq, h, aligned)

def _q(x):
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))


def _all_symbol_blocks(reach):
    """Every possible run of (2*reach+1) consecutive QPSK symbols; returns I, Q (flattened)."""
    n_sym = 2 * reach + 1
    idx = np.arange(4 ** n_sym)
    code = np.stack([(idx // 4 ** k) % 4 for k in range(n_sym)], axis=1)
    return (1 - 2 * (code & 1)).reshape(-1), (1 - 2 * (code >> 1)).reshape(-1)


def ber_semi_analytic(h, ebn0_db):
    """
    Exact BER of  mixer -> FIR h -> integrate-and-dump(16) -> sign decision  on AWGN,
    for Gray-coded QPSK.  h=None means "no filter" (pure integrate-and-dump).

    How (two ingredients):
      * SIGNAL part.  The decision value for one symbol depends only on that symbol and its
        neighbours (the FIR only reaches a few samples).  So we run the noiseless chain
        once over EVERY possible neighbourhood and read off the decision value d.
      * NOISE part.  Noise is white, so the decision noise is a weighted sum of independent
        Gaussians; its variance has a closed form from the kernel g = box16 (*) h.
    Then  P(error) = average over neighbourhoods of  Q( d / sigma_d ).
    """
    d_delay = 0 if h is None else group_delay_samples(h)
    reach = max(1, int(np.ceil(d_delay / SPS)))
    if reach > 2:
        raise ValueError("filter too long for the exact enumeration (order > 64)")

    # --- signal part: noiseless chain on all neighbourhoods ---
    I, Q = _all_symbol_blocks(reach)
    n = len(I) * SPS
    t = np.arange(n)
    cos = np.cos(2 * np.pi * FC_OVER_FS * t)
    sin = np.sin(2 * np.pi * FC_OVER_FS * t)
    s = np.repeat(I, SPS) * cos - np.repeat(Q, SPS) * sin          # locked TX equation
    if h is None:
        xi, xq = downconvert(s, cos, sin)
        yi, yq = xi, xq
    else:
        yi, yq = rx_frontend(s, cos, sin, h, aligned=True)
    dI = yi.reshape(-1, SPS).sum(axis=1)
    dQ = yq.reshape(-1, SPS).sum(axis=1)
    centre = np.arange(len(I) // (2 * reach + 1)) * (2 * reach + 1) + reach
    margin_i = (I[centre] * dI[centre])
    margin_q = (Q[centre] * dQ[centre])

    # --- noise part: exact variance of the decision value (per unit of sigma^2) ---
    g = np.ones(SPS) if h is None else np.convolve(np.ones(SPS), h)
    tt = np.arange(len(g)) - d_delay                                 # time relative to symbol start
    var_i = np.sum(g ** 2 * np.cos(2 * np.pi * FC_OVER_FS * tt) ** 2)
    var_q = np.sum(g ** 2 * np.sin(2 * np.pi * FC_OVER_FS * tt) ** 2)

    sigma = np.sqrt(channel.noise_sigma2(ebn0_db, SPS, 1.0))         # same formula Part B uses
    ber_i = _q(margin_i / (sigma * np.sqrt(var_i))).mean()
    ber_q = _q(margin_q / (sigma * np.sqrt(var_q))).mean()
    return 0.5 * (ber_i + ber_q)


def ber_penalty_db(h, ebn0_db):
    """
    How many dB of Eb/N0 the filter costs: find the Eb/N0 at which ideal QPSK theory gives the
    same BER as this receiver, then take the difference.  0 dB = no loss.
    """
    ber = ber_semi_analytic(h, ebn0_db)
    eff = brentq(lambda e: channel.theory_ber(e) - ber, -10.0, 30.0)
    return ebn0_db - eff

def _local_carrier_float(n):
    t = np.arange(n)
    return np.cos(2 * np.pi * FC_OVER_FS * t), np.sin(2 * np.pi * FC_OVER_FS * t)


def _local_carrier_fixed(n):
    """Part A's fixed-point NCO. With phase_inc = 8192 it repeats every 8 samples, so one
    period (16 samples) tiled to length n is identical to running the NCO for n samples."""
    from fixed_point import tx_nco
    c, s = tx_nco.nco_fixed(SPS)
    reps = int(np.ceil(n / SPS))
    return np.tile(c, reps)[:n], np.tile(s, reps)[:n]


def _tx_float(bits):
    """Part A's TX in float: bits -> (I,Q) -> s, plus the LO arrays."""
    from floating_point import tx_nco
    I, Q = tx_nco.map_qpsk(bits)
    cos, sin = tx_nco.nco_float(len(I) * SPS, FC_OVER_FS)
    return I, Q, tx_nco.modulate(I, Q, cos, sin, SPS), cos, sin


def _id_decisions(yi, yq, delay=0):
    """STAND-IN for Part D's integrate-and-dump + decision (only used to test this file)."""
    yi = yi[delay:]
    yq = yq[delay:]
    n_sym = len(yi) // SPS
    di = yi[:n_sym * SPS].reshape(n_sym, SPS).sum(axis=1)
    dq = yq[:n_sym * SPS].reshape(n_sym, SPS).sum(axis=1)
    return di, dq


def _ber(I, Q, di, dq):
    n = min(len(I), len(di))
    errs = np.sum((di[:n] < 0) != (I[:n] < 0)) + np.sum((dq[:n] < 0) != (Q[:n] < 0))
    return errs / (2 * n)

