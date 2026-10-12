"""
rx_frontend.py
Part C of the QPSK FPGA project golden model: the RECEIVER FRONT END.

Owns:
    - RX mixer (down-conversion), float + fixed-point -> downconvert(), downconvert_fixed()
    - FIR low-pass design (order / window / cutoff)    -> design_lpf(), check_mask()
    - FIR filtering, float + fixed-point               -> lpf(), lpf_fixed(), rx_frontend(), rx_frontend_fixed()
    - Coefficient quantization                         -> quantize_coeffs()
    - Analysis: frequency response, pole-zero, group delay, BER cost of the filter
    - Hex export of coefficients + test vectors for the Verilog testbench

Author: Aditya (Part C)
AI-assistance note: first draft written with Claude (Sonnet 5.5, Anthropic) as allowed by the
    EE2000 AI-usage policy. Every design number in this file is backed by a self-test below;
    the author must be able to explain and defend all of it in the viva.

---------------------------------------------------------------------------------------
Where this block sits (signal flow)
---------------------------------------------------------------------------------------
    Part A: bits -> (I,Q) -> s[n] = I*cos[n] - Q*sin[n]
    Part B: r[n] = s[n] + noise
    Part C: r[n] --x cos--> xi --FIR LPF--> yi        (THIS FILE)
            r[n] --x -sin-> xq --FIR LPF--> yq
    Part D: integrate-and-dump over 16 samples -> decide signs -> bits -> BER

Interface contract (matches the team plan; do not change without telling Part D):
    downconvert(r, cos, sin)          -> (xi, xq)
    design_lpf(order, fc, window)     -> h              (float taps, DC gain = 1)
    lpf(x, h)                         -> y              (causal, same length as x)
    rx_frontend(r, cos, sin, h)       -> (yi, yq)       (convenience: mixer + both filters)

---------------------------------------------------------------------------------------
Conventions that Part D MUST know  (all locked by tests at the bottom of this file)
---------------------------------------------------------------------------------------
 1. Q sign.  The TX is s = I*cos - Q*sin, so the mixer for Q uses  -sin:
        xi =  r*cos   ->  baseband I/2 (+ junk at 2*fc)
        xq = -r*sin   ->  baseband Q/2 (+ junk at 2*fc)
    The minus sign is ALREADY INSIDE downconvert(). Part A's self-loopback test and Part B's
    quick BER test use the same convention. If Part D flips Q again, every Q bit inverts.
 2. Amplitude.  The mixer is a plain multiply, so the baseband level is +/-0.5, not +/-1.
    (cos*cos averages to 1/2.) Multiply by 2 if you want the constellation at +/-1.
 3. Delay.   lpf() is causal, like the hardware shift register. A linear-phase FIR with
    order+1 taps delays everything by  D = order/2  samples (GROUP_DELAY_SAMPLES).
    The decision window for symbol k is therefore  [16k + D, 16k + D + 16)  -> timing_offset = D.
    Or call with aligned=True and use timing_offset = 0.
 4. Edges.   The very first symbol sees zeros "before" the stream and the last symbol needs D
    samples that were never sent. Skip symbol 0 (and the last one) when counting BER.
 5. Cost.    Putting a FIR in front of integrate-and-dump smears symbol edges, so it COSTS
    Eb/N0: about 0.2 dB at 0 dB, 0.25 dB at 4 dB, 0.4 dB at 10 dB (see ber_penalty_db).
    Integrate-and-dump alone already nulls the 2*fc junk exactly, so for BER the FIR buys
    nothing; it is there for a clean baseband waveform and for the Costas loop arms later.
    The BER sweep is the place to decide whether the baseline keeps it.
"""

import os
import sys

import numpy as np
from scipy import signal
from scipy.optimize import brentq
from scipy.special import erfc
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------
# Make the team's other files importable whether they sit side by side (flat folder)
# or in the repo layout  codes/Part A, codes/Part B, codes/Part C.
# ---------------------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "Part A"), os.path.join(_HERE, "..", "Part B")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

import params        # Part A: shared parameters
import fixedpoint    # Part B: shared quantizer helpers (to_int, clip_fraction, write_hex)
import channel       # Part B: AWGN channel + theoretical BER


# =======================================================================================
# 0. Part C parameters
#    (kept here so we do not edit the shared params.py; if the team later copies these
#     names into params.py, those values win automatically through getattr)
# =======================================================================================
SPS = params.SAMPLES_PER_SYMBOL                 # 16 samples per symbol
FC_OVER_FS = params.FC_OVER_FS                  # 0.125 carrier cycles per sample

FIR_ORDER = getattr(params, "FIR_ORDER", 24)    # order = taps - 1  (24 -> 25 taps)
FIR_FC = getattr(params, "FIR_FC", 0.125)       # -6 dB cutoff, in cycles per sample
FIR_WINDOW = getattr(params, "FIR_WINDOW", "hamming")
COEF_BITS = getattr(params, "COEF_BITS", 12)    # signed, format Q1.(COEF_BITS-1)
RX_IN_BITS = getattr(params, "RX_IN_BITS", 12)  # receiver input width (the "ADC" word)
RX_FULL_SCALE = getattr(params, "RX_FULL_SCALE", 8.0)   # float amplitude that maps to full scale

GROUP_DELAY_SAMPLES = FIR_ORDER // 2            # 12 samples for the default design

# Frequency "mask" the filter must satisfy (cycles/sample).
#   The wanted data lives below the symbol rate:           f <= 1/16        = 0.0625
#   The unwanted mixer product is a lobe centred on 2*fc:  f >= 2*fc - 1/16 = 0.1875
PASS_EDGE = 1.0 / SPS
STOP_EDGE = 2.0 * FC_OVER_FS - 1.0 / SPS
MASK_PASS_RIPPLE_DB = 0.5        # passband must stay within +/-0.5 dB
MASK_STOP_DB = -35.0             # stopband must be at least 35 dB down


# =======================================================================================
# 1. RX mixer (down-conversion)
# =======================================================================================

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


def _sat(x, bits):
    """Saturate an integer array to a signed `bits`-wide range (what a hardware saturator does)."""
    hi = (1 << (bits - 1)) - 1
    lo = -(1 << (bits - 1))
    return np.clip(x, lo, hi)


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


# =======================================================================================
# 2. FIR low-pass: design, analysis, filtering
# =======================================================================================

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


# =======================================================================================
# 3. Fixed-point FIR
# =======================================================================================

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


# =======================================================================================
# 4. What does the filter cost in BER?  (exact calculation, no random simulation)
# =======================================================================================

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


# =======================================================================================
# 5. Verilog hand-off: coefficient + test-vector export
# =======================================================================================

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


# =======================================================================================
# 6. Self-test / verification suite
# =======================================================================================

def _local_carrier_float(n):
    t = np.arange(n)
    return np.cos(2 * np.pi * FC_OVER_FS * t), np.sin(2 * np.pi * FC_OVER_FS * t)


def _local_carrier_fixed(n):
    """Part A's fixed-point NCO. With phase_inc = 8192 it repeats every 8 samples, so one
    period (16 samples) tiled to length n is identical to running the NCO for n samples."""
    import tx_nco
    c, s = tx_nco.nco_fixed(SPS)
    reps = int(np.ceil(n / SPS))
    return np.tile(c, reps)[:n], np.tile(s, reps)[:n]


def _tx_float(bits):
    """Part A's TX in float: bits -> (I,Q) -> s, plus the LO arrays."""
    import tx_nco
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


if __name__ == "__main__":

    print("=" * 74)
    print("Part C self-test: rx_frontend.py")
    print("=" * 74)

    h_def = design_lpf()
    rng = np.random.default_rng(2026)

    # -----------------------------------------------------------------------------
    # TEST 1: the default filter is a proper linear-phase low-pass
    # -----------------------------------------------------------------------------
    print("\n--- Test 1: default filter structure ---")
    print(f"order={FIR_ORDER}  taps={len(h_def)}  fc={FIR_FC} cycles/sample  window={FIR_WINDOW}")
    assert len(h_def) == FIR_ORDER + 1
    assert np.allclose(h_def, h_def[::-1]), "taps are not symmetric -> not linear phase"
    assert abs(h_def.sum() - 1.0) < 1e-12, "DC gain is not 1"
    assert group_delay_samples(h_def) == GROUP_DELAY_SAMPLES
    for bad in (dict(order=23), dict(fc=0.6)):
        try:
            design_lpf(**bad)
            raise AssertionError(f"design_lpf accepted bad input {bad}")
        except ValueError:
            pass
    print(f"PASS: symmetric, DC gain 1, delay = {GROUP_DELAY_SAMPLES} samples, bad inputs rejected.")

    # -----------------------------------------------------------------------------
    # TEST 2: frequency-response mask
    # -----------------------------------------------------------------------------
    print("\n--- Test 2: frequency-response mask ---")
    m = check_mask(h_def)
    print(f"data band  f<={PASS_EDGE:.4f}: gain between {m['pass_min']:+.2f} and {m['pass_max']:+.2f} dB "
          f"(mask: within +/-{MASK_PASS_RIPPLE_DB} dB)")
    print(f"junk band  f>={STOP_EDGE:.4f}: worst gain {m['stop_max']:.1f} dB (mask: <= {MASK_STOP_DB} dB)")
    assert m["pass_min"] > -MASK_PASS_RIPPLE_DB and m["pass_max"] < MASK_PASS_RIPPLE_DB
    assert m["stop_max"] < MASK_STOP_DB
    print("PASS: default design meets the mask.")

    # -----------------------------------------------------------------------------
    # TEST 3: constant group delay (the reason we chose FIR)
    # -----------------------------------------------------------------------------
    print("\n--- Test 3: group delay is constant = order/2 ---")
    f_gd = np.linspace(0.001, 0.12, 200)
    _, gd = signal.group_delay((h_def, [1.0]), w=f_gd, fs=1.0)
    print(f"group delay over the data band: min={gd.min():.6f}  max={gd.max():.6f}  (expect {GROUP_DELAY_SAMPLES})")
    assert np.allclose(gd, GROUP_DELAY_SAMPLES, atol=1e-6)
    imp = np.zeros(64)
    imp[0] = 1.0
    assert np.argmax(lpf(imp, h_def)) == GROUP_DELAY_SAMPLES, "impulse peak not at D (causal mode)"
    assert np.argmax(lpf(imp, h_def, aligned=True)) == 0, "impulse peak not at 0 (aligned mode)"
    print("PASS: every frequency is delayed by the same 12 samples; aligned mode removes it.")

    print("\n--- Test 3b: the timing_offset Part D must use ---")
    n3 = 300
    bits3 = rng.integers(0, 2, 2 * n3).astype(np.uint8)
    I3, Q3, s3, c3, sn3 = _tx_float(bits3)
    r3 = channel.awgn(s3, 6.0, mode="numpy", seed=9)
    yi_c, yq_c = rx_frontend(r3, c3, sn3, h_def, aligned=False)          # causal, hardware-like
    yi_a, yq_a = rx_frontend(r3, c3, sn3, h_def, aligned=True)           # delay removed
    di_c, _ = _id_decisions(yi_c, yq_c, delay=GROUP_DELAY_SAMPLES)       # window starts at 16k + D
    di_a, _ = _id_decisions(yi_a, yq_a, delay=0)
    n_cmp = len(di_c)                                                    # the last symbol is incomplete in causal mode
    assert n_cmp >= n3 - 1
    assert np.allclose(di_c[:n_cmp], di_a[:n_cmp]), "timing_offset = D does not match aligned mode"
    print(f"causal output + timing_offset={GROUP_DELAY_SAMPLES} gives the same decision values as "
          f"aligned output + timing_offset=0 ({n_cmp} of {n3} symbols; the last one is cut off).")
    print("PASS.")

    # -----------------------------------------------------------------------------
    # TEST 4: pole-zero structure
    # -----------------------------------------------------------------------------
    print("\n--- Test 4: pole-zero structure ---")
    zeros, n_inf = finite_zeros(h_def)
    print(f"zeros: {len(zeros)} finite + {n_inf} at infinity (= order {FIR_ORDER}); "
          f"poles: {FIR_ORDER}, all at z = 0 (no feedback => always stable)")
    assert len(zeros) + n_inf == FIR_ORDER and np.all(np.abs(zeros) < 1e3)
    print("PASS.")

    # -----------------------------------------------------------------------------
    # TEST 5: mixer sign + amplitude, locked against Part A's TX
    # -----------------------------------------------------------------------------
    print("\n--- Test 5: mixer recovers (I/2, Q/2) from Part A's transmitter ---")
    import tx_nco
    bits5 = tx_nco.gen_bits(400)
    I5, Q5, s5, c5, sn5 = _tx_float(bits5)
    xi5, xq5 = downconvert(s5, c5, sn5)
    mean_i = xi5.reshape(-1, SPS).mean(axis=1)
    mean_q = xq5.reshape(-1, SPS).mean(axis=1)
    err = np.max(np.abs(mean_i - I5 / 2)) + np.max(np.abs(mean_q - Q5 / 2))
    print(f"max |mean(xi) - I/2| + |mean(xq) - Q/2| over a symbol = {err:.2e}")
    assert err < 1e-12, "mixer sign/amplitude convention broken"
    print("PASS: Q uses -sin inside downconvert(); baseband level is +/-0.5.")

    # -----------------------------------------------------------------------------
    # TEST 6: after the filter the 2*fc junk is gone (steady symbol, no transitions)
    # -----------------------------------------------------------------------------
    print("\n--- Test 6: junk removal on a steady symbol ---")
    n6 = 400 * SPS
    I6 = np.ones(400)
    Q6 = -np.ones(400)
    c6, s6 = _local_carrier_float(n6)
    s_tx6 = np.repeat(I6, SPS) * c6 - np.repeat(Q6, SPS) * s6
    yi6, yq6 = rx_frontend(s_tx6, c6, s6, h_def)
    settled = slice(len(h_def), n6)
    rip_i = np.max(np.abs(yi6[settled] - 0.5))
    rip_q = np.max(np.abs(yq6[settled] + 0.5))
    raw_i = np.max(np.abs(downconvert(s_tx6, c6, s6)[0][settled] - 0.5))
    print(f"ripple around +/-0.5 BEFORE filter: {raw_i:.4f} | AFTER filter: I {rip_i:.2e}, Q {rip_q:.2e}")
    assert rip_i < 0.01 and rip_q < 0.01 and raw_i > 0.3
    print("PASS: junk of size ~0.5 before the filter is squeezed below 0.01 after it.")

    # -----------------------------------------------------------------------------
    # TEST 7: order / window trade-off table (this is a table for the report)
    # -----------------------------------------------------------------------------
    print("\n--- Test 7: order / window trade-off at fc = 0.125, Eb/N0 = 4 dB ---")
    print(f"{'window':9s} {'order':>5s} {'taps':>4s} {'mults*':>6s} {'delay':>5s} "
          f"{'droop@Rs':>9s} {'stopband':>9s} {'BER cost':>9s}")
    table = {}
    for window in ["hamming", "hann", "blackman"]:
        for order in [8, 12, 16, 24, 30]:
            hh = design_lpf(order, 0.125, window)
            mm = check_mask(hh)
            pen = ber_penalty_db(hh, 4.0)
            table[(window, order)] = (mm, pen)
            print(f"{window:9s} {order:5d} {order+1:4d} {order//2+1:6d} {order//2:5d} "
                  f"{mm['pass_min']:8.2f}dB {mm['stop_max']:8.1f}dB {pen:8.3f}dB")
    print("  * mults = multipliers per branch if the symmetric taps are paired (pre-adder trick).")
    pens_24 = [table[(w, 24)][1] for w in ["hamming", "hann", "blackman"]]
    assert max(pens_24) - min(pens_24) < 0.05, "BER cost should barely depend on the window"
    assert table[("hamming", 8)][0]["stop_max"] > table[("hamming", 24)][0]["stop_max"], \
        "longer filter should reject more"
    print("PASS: longer order -> deeper stopband; the BER cost hardly changes (it is set by fc).")

    # -----------------------------------------------------------------------------
    # TEST 8: BER cost vs cutoff (shows WHY fc = 0.125)
    # -----------------------------------------------------------------------------
    print("\n--- Test 8: BER cost vs cutoff, hamming order 24, Eb/N0 = 4 dB ---")
    fc_list = np.array([0.07, 0.09, 0.11, 0.125, 0.15, 0.18, 0.22, 0.30])
    pen_fc = []
    for fcv in fc_list:
        hh = design_lpf(24, fcv, "hamming")
        pen_fc.append(ber_penalty_db(hh, 4.0))
        print(f"  fc={fcv:5.3f}: BER cost {pen_fc[-1]:6.3f} dB | stopband {check_mask(hh)['stop_max']:7.1f} dB")
    assert all(np.diff(pen_fc) < 1e-3), "BER cost should shrink as the cutoff widens"
    print("PASS: narrow cutoff smears more -> costs more; wide cutoff rejects less -> fc = 0.125 balances both.")

    # -----------------------------------------------------------------------------
    # TEST 9: exact calculation vs Monte-Carlo (validates ber_semi_analytic)
    # -----------------------------------------------------------------------------
    print("\n--- Test 9: exact BER vs Monte-Carlo (numpy noise, 150k symbols) ---")
    n_sym = 150_000
    bits9 = rng.integers(0, 2, 2 * n_sym).astype(np.uint8)
    I9, Q9, s9, c9, sn9 = _tx_float(bits9)
    for ebn0 in (2.0, 4.0):
        r9 = channel.awgn(s9, ebn0, mode="numpy", seed=17)
        yi9, yq9 = rx_frontend(r9, c9, sn9, h_def, aligned=True)
        ber_mc = _ber(I9, Q9, *_id_decisions(yi9, yq9))
        ber_ex = ber_semi_analytic(h_def, ebn0)
        ber_id = ber_semi_analytic(None, ebn0)
        print(f"  Eb/N0={ebn0} dB: theory {channel.theory_ber(ebn0):.5f} | no-filter exact {ber_id:.5f} | "
              f"filtered exact {ber_ex:.5f} | filtered Monte-Carlo {ber_mc:.5f}")
        assert abs(ber_id - channel.theory_ber(ebn0)) / channel.theory_ber(ebn0) < 1e-6, \
            "pure integrate-and-dump must equal textbook QPSK theory"
        assert abs(ber_mc - ber_ex) / ber_ex < 0.06, "Monte-Carlo disagrees with the exact calculation"
    print("PASS: no-filter case equals theory exactly; filtered case matches simulation within noise.")

    # -----------------------------------------------------------------------------
    # TEST 10: coefficient quantization
    # -----------------------------------------------------------------------------
    print("\n--- Test 10: coefficient word-length sweep ---")
    print(f"{'bits':>4s} {'stopband':>9s} {'BER cost':>9s} {'max|h_int|':>10s}")
    stop_by_bits = {}
    for b in (4, 6, 8, 10, 12, 14, 16):
        hq = coeffs_to_float(quantize_coeffs(h_def, b), b)
        stop_by_bits[b] = check_mask(hq)["stop_max"]
        print(f"{b:4d} {stop_by_bits[b]:8.1f}dB {ber_penalty_db(hq, 4.0):8.3f}dB "
              f"{int(np.abs(quantize_coeffs(h_def, b)).max()):10d}")
    h_int = quantize_coeffs(h_def, COEF_BITS)
    assert h_int.sum() == 1 << (COEF_BITS - 1), "quantized DC gain is not exactly 1"
    assert np.array_equal(h_int, h_int[::-1]), "quantization broke the symmetry"
    assert stop_by_bits[COEF_BITS] < MASK_STOP_DB, "chosen coefficient width misses the mask"
    assert abs(ber_penalty_db(coeffs_to_float(h_int, COEF_BITS), 4.0) - ber_penalty_db(h_def, 4.0)) < 0.005
    print(f"PASS: {COEF_BITS}-bit taps keep DC gain exactly 1, stay symmetric, meet the mask, and add no BER cost.")
    print(f"      accumulator needs {acc_bits_needed(h_int)} bits (a DSP48E1 accumulator has 48).")

    # -----------------------------------------------------------------------------
    # TEST 11: fixed-point chain vs float chain
    # -----------------------------------------------------------------------------
    print("\n--- Test 11: fixed-point front end vs float front end (4 dB) ---")
    n_sym = 100_000
    bits11 = rng.integers(0, 2, 2 * n_sym).astype(np.uint8)
    I11, Q11, s11, c11, sn11 = _tx_float(bits11)
    r11 = channel.awgn(s11, 4.0, mode="numpy", seed=5)
    r_int = quantize_rx(r11)
    clip = fixedpoint.clip_fraction(r11, RX_IN_BITS, RX_FULL_SCALE)
    c_int, s_int = _local_carrier_fixed(len(r_int))
    yi_fx, yq_fx = rx_frontend_fixed(r_int, c_int, s_int, h_int, aligned=True)
    yi_fl, yq_fl = rx_frontend(r11, c11, sn11, coeffs_to_float(h_int, COEF_BITS), aligned=True)
    lsb = RX_FULL_SCALE / (1 << (RX_IN_BITS - 1))          # one code in float units
    diff = np.concatenate([yi_fx * lsb - yi_fl, yq_fx * lsb - yq_fl]) / lsb
    print(f"input clipping fraction: {clip:.2e} | fixed-vs-float error: rms {np.sqrt(np.mean(diff**2)):.2f} LSB, "
          f"max {np.max(np.abs(diff)):.1f} LSB (1 LSB = {lsb:.4f} in TX-amplitude units)")
    assert clip < 1e-3
    assert np.sqrt(np.mean(diff ** 2)) < 2.0 and np.max(np.abs(diff)) < 8.0
    ber_fx = _ber(I11, Q11, *_id_decisions(yi_fx, yq_fx))
    ber_fl = _ber(I11, Q11, *_id_decisions(yi_fl, yq_fl))
    print(f"BER float chain {ber_fl:.5f} | BER fixed chain {ber_fx:.5f}")
    assert abs(ber_fx - ber_fl) / ber_fl < 0.03, "fixed-point chain changes the BER"
    print("PASS: integer chain tracks the float chain to ~1 LSB and gives the same BER.")

    # -----------------------------------------------------------------------------
    # TEST 12: integer overflow / saturation behaves
    # -----------------------------------------------------------------------------
    print("\n--- Test 12: saturation ---")
    big = np.full(200, (1 << (RX_IN_BITS - 1)) - 1, dtype=np.int64)       # constant at +full scale
    y_big = lpf_fixed(big, h_int)
    assert y_big.max() <= (1 << (RX_IN_BITS - 1)) - 1 and y_big.min() >= -(1 << (RX_IN_BITS - 1))
    print(f"full-scale DC in -> settles to {y_big[-1]} (max code {(1 << (RX_IN_BITS - 1)) - 1}); never wraps.")
    print("PASS.")

    # -----------------------------------------------------------------------------
    # TEST 13: run the whole chain with Part B's LFSR (hardware-style) noise
    # -----------------------------------------------------------------------------
    print("\n--- Test 13: LFSR noise smoke test (Part B hardware-style noise, 4000 symbols) ---")
    n_sym = 4000
    bits13 = tx_nco.gen_bits(2 * n_sym)
    I13, Q13, s13, c13, sn13 = _tx_float(bits13)
    r13 = channel.awgn(s13, 2.0, mode="lfsr", seed=3)
    yi13, yq13 = rx_frontend(r13, c13, sn13, h_def, aligned=True)
    ber13 = _ber(I13, Q13, *_id_decisions(yi13, yq13))
    ex13 = ber_semi_analytic(h_def, 2.0)
    print(f"BER with LFSR noise at 2 dB: {ber13:.4f} | exact prediction {ex13:.4f}")
    assert abs(ber13 - ex13) / ex13 < 0.20, "LFSR-noise chain is far from prediction"
    print("PASS: chain works with the hardware-style noise source too.")

    # -----------------------------------------------------------------------------
    # TEST 14: hex export for Verilog testbench
    # -----------------------------------------------------------------------------
    print("\n--- Test 14: export test vectors ---")
    xi_v, xq_v, yi_v, yq_v = export_rx_vectors("rx_vectors", h_int, r_int, c_int, s_int)
    back = np.array([int(l, 16) for l in open(os.path.join("rx_vectors", "fir_coeffs.hex")).read().split()])
    back = np.where(back >= 1 << (COEF_BITS - 1), back - (1 << COEF_BITS), back)
    assert np.array_equal(back, h_int), "fir_coeffs.hex round-trip failed"
    print(f"wrote rx_vectors/*.hex  (taps: {len(h_int)}, vectors: {len(xi_v)} samples each)")
    print("PASS: coefficient file round-trips.")

    # -----------------------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    def shade(a):
        a.axvspan(0, PASS_EDGE, color="tab:green", alpha=0.12, label="data band")
        a.axvspan(STOP_EDGE, 0.5, color="tab:red", alpha=0.10, label="junk lobe (2fc)")

    # (a) window comparison at the same order
    a = ax[0, 0]
    shade(a)
    for wn, col in [("boxcar", "gray"), ("hamming", "tab:blue"), ("hann", "tab:orange"), ("blackman", "tab:purple")]:
        f_, H_ = freq_response(design_lpf(24, 0.125, wn))
        a.plot(f_, 20 * np.log10(np.abs(H_) + 1e-12), color=col, label=wn)
    a.axhline(MASK_STOP_DB, color="k", ls=":", lw=0.8)
    a.set_ylim(-110, 5)
    a.set_title("Windows at order 24, fc = 0.125")
    a.set_xlabel("frequency (cycles/sample)")
    a.set_ylabel("gain (dB)")
    a.legend(fontsize=7)
    a.grid(True)

    # (b) order comparison, hamming
    a = ax[0, 1]
    shade(a)
    for od in (8, 16, 24, 30):
        f_, H_ = freq_response(design_lpf(od, 0.125, "hamming"))
        a.plot(f_, 20 * np.log10(np.abs(H_) + 1e-12), label=f"order {od}")
    a.axhline(MASK_STOP_DB, color="k", ls=":", lw=0.8)
    a.set_ylim(-110, 5)
    a.set_title("Orders (hamming, fc = 0.125)")
    a.set_xlabel("frequency (cycles/sample)")
    a.set_ylabel("gain (dB)")
    a.legend(fontsize=7)
    a.grid(True)

    # (c) quantized vs float response
    a = ax[0, 2]
    shade(a)
    f_, H_ = freq_response(h_def)
    a.plot(f_, 20 * np.log10(np.abs(H_) + 1e-12), "k", lw=2, label="float")
    for b in (6, 8, 12):
        f_, H_ = freq_response(coeffs_to_float(quantize_coeffs(h_def, b), b))
        a.plot(f_, 20 * np.log10(np.abs(H_) + 1e-12), label=f"{b}-bit taps")
    a.axhline(MASK_STOP_DB, color="k", ls=":", lw=0.8)
    a.set_ylim(-110, 5)
    a.set_title("Coefficient quantization")
    a.set_xlabel("frequency (cycles/sample)")
    a.set_ylabel("gain (dB)")
    a.legend(fontsize=7)
    a.grid(True)

    # (d) pole-zero
    a = ax[1, 0]
    th = np.linspace(0, 2 * np.pi, 400)
    a.plot(np.cos(th), np.sin(th), "gray", lw=0.8)
    a.plot(zeros.real, zeros.imag, "o", mfc="none", color="tab:blue",
           label=f"{len(zeros)} zeros (+{n_inf} at infinity)")
    a.plot([0], [0], "x", color="tab:red", ms=10, label=f"{FIR_ORDER} poles at z=0")
    a.set_xlim(-2.5, 2.5)
    a.set_ylim(-2.5, 2.5)
    a.set_aspect("equal")
    a.set_title("Pole-zero plot (default filter)")
    a.set_xlabel("Re")
    a.set_ylabel("Im")
    a.legend(fontsize=7, loc="lower left")
    a.grid(True)

    # (e) group delay: FIR vs IIR
    a = ax[1, 1]
    f_g = np.linspace(0.001, 0.25, 400)
    _, gd_fir = signal.group_delay((h_def, [1.0]), w=f_g, fs=1.0)
    b_iir, a_iir = signal.butter(4, FIR_FC, fs=1.0)
    _, gd_iir = signal.group_delay((b_iir, a_iir), w=f_g, fs=1.0)
    a.plot(f_g, gd_fir, label=f"FIR (flat = {GROUP_DELAY_SAMPLES} samples)")
    a.plot(f_g, gd_iir, label="4th-order Butterworth IIR, same fc")
    a.axvspan(0, PASS_EDGE, color="tab:green", alpha=0.12)
    a.set_title("Group delay: why FIR")
    a.set_xlabel("frequency (cycles/sample)")
    a.set_ylabel("delay (samples)")
    a.legend(fontsize=7)
    a.grid(True)

    # (f) BER cost vs cutoff for several orders
    a = ax[1, 2]
    fc_scan = np.linspace(0.07, 0.30, 24)
    for od in (8, 16, 24, 30):
        a.plot(fc_scan, [ber_penalty_db(design_lpf(od, f, "hamming"), 4.0) for f in fc_scan], label=f"order {od}")
    a.axvline(FIR_FC, color="k", ls=":", lw=0.8)
    a.set_title("BER cost of the filter (Eb/N0 = 4 dB)")
    a.set_xlabel("cutoff fc (cycles/sample)")
    a.set_ylabel("Eb/N0 lost (dB)")
    a.legend(fontsize=7)
    a.grid(True)

    plt.tight_layout()
    plt.savefig("part_c_verification_plots.png", dpi=150)

    # ---- second figure: end-to-end results for the report ----
    fig2, bx = plt.subplots(1, 3, figsize=(17, 5))

    ax0 = bx[0]
    ax0.stem(np.arange(len(h_def)), h_def, linefmt="C0-", markerfmt="C0o", basefmt="k-", label="float")
    ax0.plot(np.arange(len(h_def)), coeffs_to_float(h_int, COEF_BITS), "r.", label=f"{COEF_BITS}-bit")
    ax0.set_title("Impulse response (the 25 taps)")
    ax0.set_xlabel("tap k")
    ax0.set_ylabel("h[k]")
    ax0.legend(fontsize=8)
    ax0.grid(True)

    ax1 = bx[1]
    eb_axis = np.arange(0, 10.01, 0.5)
    ax1.semilogy(eb_axis, [channel.theory_ber(e) for e in eb_axis], "k-", label="QPSK theory")
    ax1.semilogy(eb_axis, [ber_semi_analytic(None, e) for e in eb_axis], "g--", label="mixer + integrate-and-dump (no FIR)")
    ax1.semilogy(eb_axis, [ber_semi_analytic(h_def, e) for e in eb_axis], "b-", label="mixer + FIR + integrate-and-dump")
    ax1.set_title("BER vs Eb/N0 (exact)")
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("BER")
    ax1.set_ylim(1e-6, 1)
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both")

    ax2 = bx[2]
    r_c = channel.awgn(s9, 6.0, mode="numpy", seed=1)
    yi_c, yq_c = rx_frontend(r_c, c9, sn9, h_def, aligned=True)
    di_c, dq_c = _id_decisions(yi_c, yq_c)
    k = slice(1, 3001)
    ax2.scatter(di_c[k] / 8.0, dq_c[k] / 8.0, s=3, alpha=0.5)       # /8: sum of 16 samples of +/-0.5
    ax2.set_title("Constellation after mixer + FIR + I&D (6 dB)")
    ax2.set_xlabel("I")
    ax2.set_ylabel("Q")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(0, color="gray", lw=0.5)
    ax2.set_aspect("equal")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("part_c_endtoend_plots.png", dpi=150)
    print("\nPlots saved to part_c_verification_plots.png and part_c_endtoend_plots.png")
    try:
        plt.show()
    except Exception as e:
        print(f"(plt.show() skipped -- no display available: {e})")

    print("\n" + "=" * 74)
    print("All Part C self-tests completed and PASSED (every check above is an assert).")
    print("=" * 74)
