"""
tx_nco.py
Part A of the QPSK FPGA project golden model.

Owns:
    - PRBS bit source               -> gen_bits()
    - Gray-coded QPSK bit mapper    -> map_qpsk()
    - NCO, float + fixed-point      -> nco_float(), nco_fixed()
    - TX mixer, float + fixed-point -> modulate(), modulate_fixed()
    - LUT hex export for Verilog    -> export_lut_hex()

Author: [FILL IN YOUR NAME]
AI-assistance note: [FILL IN per course AI-usage policy]

Interface contract (do not change without notifying Parts B/C/D):
    gen_bits(n, seed=...)                 -> bits : np.ndarray of 0/1
    map_qpsk(bits)                        -> (I, Q), values +/-1
    nco_float(n_samples, fc_over_fs)      -> (cos, sin), float in [-1,1]
    nco_fixed(n_samples, phase_inc, ...)  -> (cos, sin), int, fixed point
    modulate(I, Q, cos, sin, sps)         -> s : float sample array
    modulate_fixed(I, Q, cos, sin, sps)   -> s : int sample array (13-bit)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import params


# =====================================================================
# 1. PRBS bit source
# =====================================================================

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


# =====================================================================
# 2. Gray-coded QPSK bit-to-symbol mapper
# =====================================================================

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
    b0 = (I < 0).astype(np.uint8)
    b1 = (Q < 0).astype(np.uint8)
    bits = np.empty(2 * len(I), dtype=np.uint8)
    bits[0::2] = b0
    bits[1::2] = b1
    return bits


# =====================================================================
# 3a. NCO -- floating point
# =====================================================================

def nco_float(n_samples, fc_over_fs=params.FC_OVER_FS, init_phase=0.0):
    """Ideal floating-point carrier generator."""
    n = np.arange(n_samples)
    phase = 2 * np.pi * fc_over_fs * n + init_phase
    return np.cos(phase), np.sin(phase)


# =====================================================================
# 3b. NCO -- fixed point (phase accumulator + LUT)
# =====================================================================

def build_nco_lut(lut_size=params.NCO_LUT_SIZE,
                   datapath_bits=params.DATAPATH_BITS):
    """Precompute the sin/cos lookup table as signed fixed-point integers."""
    full_scale = (1 << (datapath_bits - 1)) - 1
    angles = 2 * np.pi * np.arange(lut_size) / lut_size
    lut_cos = np.round(full_scale * np.cos(angles)).astype(int)
    lut_sin = np.round(full_scale * np.sin(angles)).astype(int)
    return lut_cos, lut_sin


def phase_inc_is_exact_for_lut(phase_inc, acc_bits, lut_bits):
    """
    Exact (provable) condition for zero LUT-truncation error:
    every phase value visited (multiples of phase_inc mod 2^acc_bits)
    has all its low (acc_bits-lut_bits) bits equal to zero. This holds
    for every visited multiple iff it holds for phase_inc itself, since
    divisibility by 2^shift is preserved under repeated addition.
    """
    shift = acc_bits - lut_bits
    return (phase_inc % (1 << shift)) == 0


def nco_fixed(n_samples, phase_inc=params.NCO_PHASE_INC,
              acc_bits=params.NCO_ACC_BITS, lut_bits=params.NCO_LUT_BITS,
              datapath_bits=params.DATAPATH_BITS, init_phase=0):
    """Hardware-realistic NCO: phase accumulator + quantized sin/cos LUT."""
    lut_cos, lut_sin = build_nco_lut(lut_size=(1 << lut_bits),
                                      datapath_bits=datapath_bits)
    acc_max = 1 << acc_bits
    shift = acc_bits - lut_bits
    if shift < 0:
        raise ValueError("LUT address width cannot exceed accumulator width")

    phase = init_phase % acc_max
    cos_out = np.zeros(n_samples, dtype=int)
    sin_out = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        idx = phase >> shift
        cos_out[i] = lut_cos[idx]
        sin_out[i] = lut_sin[idx]
        phase = (phase + phase_inc) % acc_max

    return cos_out, sin_out


# =====================================================================
# 4. TX Mixer / Modulator
# =====================================================================

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


def required_mixer_bits(datapath_bits):
    """
    Minimum signed bit-width needed to hold I*cos - Q*sin without
    overflow, for I,Q in {+1,-1} and cos/sin quantized to datapath_bits.
    Worst case magnitude = full_scale * sqrt(2), occurring at the
    +/-45 deg LUT entries.
    """
    full_scale = (1 << (datapath_bits - 1)) - 1
    worst_case = full_scale * np.sqrt(2)
    bits = 1
    while (1 << (bits - 1)) - 1 < worst_case:
        bits += 1
    return bits


def modulate_fixed(I, Q, cos_int, sin_int, sps=params.SAMPLES_PER_SYMBOL,
                    out_bits=params.MIXER_OUT_BITS):
    """
    Fixed-point (integer) TX mixer -- the real counterpart to modulate(),
    matching what tx_mixer.v will compute. I,Q in {+1,-1} (exact
    integers), cos_int/sin_int from nco_fixed() (exact integers), so
    the multiply-subtract is exact -- no rounding needed here, only an
    overflow check against out_bits.

    Raises OverflowError if the result does not fit in a signed
    out_bits-wide register, rather than silently wrapping/clipping,
    so overflow bugs are caught immediately instead of surfacing later
    as mysterious constellation corruption.
    """
    I = np.asarray(I).astype(int)
    Q = np.asarray(Q).astype(int)
    n_expected = len(I) * sps

    if len(cos_int) < n_expected or len(sin_int) < n_expected:
        raise ValueError("carrier arrays too short for given I/Q length and sps")

    I_up = np.repeat(I, sps)
    Q_up = np.repeat(Q, sps)
    s = I_up * cos_int[:n_expected] - Q_up * sin_int[:n_expected]

    max_val = (1 << (out_bits - 1)) - 1
    min_val = -(1 << (out_bits - 1))
    if np.any(s > max_val) or np.any(s < min_val):
        raise OverflowError(
            f"modulate_fixed output [{s.min()},{s.max()}] exceeds "
            f"{out_bits}-bit signed range [{min_val},{max_val}]"
        )
    return s


# =====================================================================
# 5. LUT hex export for Verilog $readmemh
# =====================================================================

def export_lut_hex(lut, bits, filename):
    """
    Export a signed integer LUT array to a $readmemh-compatible hex
    file. Negative numbers are stored in two's-complement form over
    `bits` bits, one value per line, as plain hex digits (no 0x prefix,
    no commas -- this is exactly what $readmemh expects).
    """
    mask = (1 << bits) - 1
    hex_width = (bits + 3) // 4   # hex digits needed to cover `bits` bits
    with open(filename, "w") as f:
        for val in lut:
            twos = int(val) & mask
            f.write(f"{twos:0{hex_width}x}\n")


def read_hex_twos_complement(filename, bits):
    """Inverse of export_lut_hex(), used only to verify the round-trip."""
    half = 1 << (bits - 1)
    full = 1 << bits
    vals = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u = int(line, 16)
            if u >= half:
                u -= full
            vals.append(u)
    return np.array(vals)


# =====================================================================
# 6. Self-test / verification suite
# =====================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Part A self-test: tx_nco.py")
    print("=" * 70)

    # -----------------------------------------------------------
    # TEST 1: PRBS bit source
    # -----------------------------------------------------------
    print("\n--- Test 1: PRBS bit source ---")
    period = (1 << params.PRBS_LEN) - 1   # 127
    test_bits = gen_bits(period * 2)
    repeats_ok = np.array_equal(test_bits[:period], test_bits[period:2*period])
    ones_frac = test_bits[:period].mean()
    expected_ones_frac = (2 ** (params.PRBS_LEN - 1)) / period

    print(f"Period check: repeats correctly = {repeats_ok} (expect True)")
    print(f"Ones fraction: {ones_frac:.4f} (expect {expected_ones_frac:.4f})")
    assert repeats_ok, "PRBS period check FAILED"
    assert abs(ones_frac - expected_ones_frac) < 1e-9, "PRBS statistics FAILED"

    b1 = gen_bits(50, seed=params.PRBS_SEED)
    b2 = gen_bits(50, seed=params.PRBS_SEED)
    assert np.array_equal(b1, b2), "PRBS not reproducible with fixed seed"
    print("PASS: period, statistics, reproducibility all correct.")

    # Test 1b: cross-check against an independently-written, bit-literal
    # RTL reference implementation (not just "same period").
    print("\n--- Test 1b: Cross-check against explicit RTL-bit reference ---")
    ref = _prbs7_rtl_reference(300, seed=params.PRBS_SEED)
    generic = gen_bits(300, seed=params.PRBS_SEED,
                        taps=params.PRBS_TAPS, reg_len=params.PRBS_LEN)
    rtl_match = np.array_equal(ref, generic)
    print(f"Generic gen_bits(taps=(1,7)) matches hardcoded RTL mirror: "
          f"{rtl_match}")
    assert rtl_match, (
        "gen_bits() does not match the documented RTL bit convention "
        "(output=lfsr[0] before shift, feedback=lfsr[0]^lfsr[6], "
        "insert at bit 6) -- Verilog will NOT match this Python model"
    )
    print("PASS: tap convention verified against literal RTL bit mapping.")

    # -----------------------------------------------------------
    # TEST 2: Gray-coded mapper
    # -----------------------------------------------------------
    print("\n--- Test 2: QPSK bit mapper (round-trip) ---")
    all_pairs = np.array([0,0, 0,1, 1,1, 1,0], dtype=np.uint8)
    I_test, Q_test = map_qpsk(all_pairs)
    for lbl, i_val, q_val in zip(["(0,0)","(0,1)","(1,1)","(1,0)"],
                                  I_test, Q_test):
        print(f"  {lbl} -> (I={i_val:+.0f}, Q={q_val:+.0f})")
    recovered = demap_qpsk_reference(I_test, Q_test)
    assert np.array_equal(recovered, all_pairs), "map/demap not inverses"
    print("PASS: round-trip consistent.")

    print("\n--- Test 2b: Verify against LOCKED params.py table ---")
    locked_table = {(0,0):(+1,+1), (0,1):(+1,-1), (1,1):(-1,-1), (1,0):(-1,+1)}
    for (b0,b1), (I_exp,Q_exp) in locked_table.items():
        I_got, Q_got = map_qpsk(np.array([b0,b1], dtype=np.uint8))
        match = (I_got[0]==I_exp) and (Q_got[0]==Q_exp)
        print(f"  ({b0},{b1}) -> got ({I_got[0]:+.0f},{Q_got[0]:+.0f}), "
              f"expected ({I_exp:+d},{Q_exp:+d})  [{'OK' if match else 'MISMATCH'}]")
        assert match, f"Mapper disagrees with locked table for ({b0},{b1})"
    print("PASS: matches locked table exactly.")

    print("\n--- Test 2c: Gray-code adjacency check ---")
    angular_order = [(0,0), (1,0), (1,1), (0,1)]
    for k in range(4):
        a, b = angular_order[k], angular_order[(k+1) % 4]
        dist = sum(x != y for x, y in zip(a, b))
        print(f"  {a} -> {b} : bit flips = {dist}")
        assert dist == 1, f"Gray property violated between {a} and {b}"
    print("PASS: Gray-code property confirmed.")

    # -----------------------------------------------------------
    # TEST 3: NCO frequency check
    # -----------------------------------------------------------
    print("\n--- Test 3: NCO frequency check ---")
    N_FFT = 4096
    cos_f, sin_f = nco_float(N_FFT, params.FC_OVER_FS)
    spectrum = np.abs(np.fft.fft(cos_f))
    freqs = np.fft.fftfreq(N_FFT)
    peak_freq = abs(freqs[np.argmax(spectrum[:N_FFT//2])])
    print(f"Expected fc/fs: {params.FC_OVER_FS} | Measured: {peak_freq:.4f}")
    assert abs(peak_freq - params.FC_OVER_FS) < (1.0/N_FFT), \
        "NCO frequency mismatch"
    print("PASS.")

    # -----------------------------------------------------------
    # TEST 4: Fixed-point NCO quantization error -- COS AND SIN
    # -----------------------------------------------------------
    print("\n--- Test 4: Fixed-point NCO error at system fc/fs (cos AND sin) ---")
    N = 256
    cos_fx, sin_fx = nco_fixed(N)
    cos_fl, sin_fl = nco_float(N, params.FC_OVER_FS)
    cos_fx_n = cos_fx / params.DATAPATH_FULL_SCALE
    sin_fx_n = sin_fx / params.DATAPATH_FULL_SCALE

    err_cos = cos_fx_n - cos_fl
    err_sin = sin_fx_n - sin_fl
    sqnr_cos = 10*np.log10(np.mean(cos_fl**2)/np.mean(err_cos**2))
    sqnr_sin = 10*np.log10(np.mean(sin_fl**2)/np.mean(err_sin**2))
    print(f"SQNR cos: {sqnr_cos:.2f} dB | SQNR sin: {sqnr_sin:.2f} dB")
    assert sqnr_cos > 50, "cos SQNR unexpectedly low at system operating point"
    assert sqnr_sin > 50, "sin SQNR unexpectedly low at system operating point"
    print("PASS: both cos and sin verified.")

    # -----------------------------------------------------------
    # TEST 5a: Precision sweep at SYSTEM fc/fs=1/8
    # Corrected understanding: NOT flat in general. The exact condition
    # for zero truncation is phase_inc % 2^(acc_bits-lut_bits) == 0,
    # not merely "phase_inc is an integer".
    # -----------------------------------------------------------
    print("\n--- Test 5a: Precision sweep at system fc/fs "
          f"(corrected condition check) ---")
    configs = [(4, 2), (8, 4), (16, 8)]
    cos_fl_sys, _ = nco_float(N, params.FC_OVER_FS)
    sqnr_5a = {}
    for acc_b, lut_b in configs:
        inc = round(params.FC_OVER_FS * (1 << acc_b))
        exact = phase_inc_is_exact_for_lut(inc, acc_b, lut_b)
        cfx, _ = nco_fixed(N, phase_inc=inc, acc_bits=acc_b, lut_bits=lut_b)
        cfx_n = cfx / params.DATAPATH_FULL_SCALE
        e = cfx_n - cos_fl_sys
        sqnr = 10*np.log10(np.mean(cos_fl_sys**2)/np.mean(e**2))
        sqnr_5a[(acc_b, lut_b)] = sqnr
        print(f"  acc={acc_b:2d}, lut={lut_b} | exact-for-LUT: {exact!s:5} "
              f"| SQNR={sqnr:6.2f} dB")
        # The divisibility condition must correctly PREDICT high vs low SQNR
        if exact:
            assert sqnr > 50, (
                f"Config {(acc_b,lut_b)} satisfies the exact-divisibility "
                f"condition but shows low SQNR -- prediction rule is wrong"
            )
        else:
            assert sqnr < 50, (
                f"Config {(acc_b,lut_b)} violates the exact-divisibility "
                f"condition but shows high SQNR -- prediction rule is wrong"
            )
    print("PASS: phase_inc % 2^shift == 0 correctly predicts truncation-free "
          "configs.")

    # -----------------------------------------------------------
    # TEST 5b-i: Generic-ratio sweep, FIXED to compare against the
    # REALIZED frequency (phase_inc / 2^acc_bits), not the unrounded
    # target. This isolates quantization/truncation error from
    # frequency-mismatch error.
    # -----------------------------------------------------------
    print("\n--- Test 5b-i: Generic-ratio sweep vs REALIZED frequency ---")
    sqnr_5b = {}
    for acc_b, lut_b in configs:
        inc = round(params.GENERIC_TEST_FC_OVER_FS * (1 << acc_b))
        realized = inc / (1 << acc_b)
        cos_fl_real, _ = nco_float(N, realized)
        cfx, _ = nco_fixed(N, phase_inc=inc, acc_bits=acc_b, lut_bits=lut_b)
        cfx_n = cfx / params.DATAPATH_FULL_SCALE
        e = cfx_n - cos_fl_real
        sqnr = 10*np.log10(np.mean(cos_fl_real**2)/np.mean(e**2))
        sqnr_5b[(acc_b, lut_b)] = sqnr
        print(f"  acc={acc_b:2d}, lut={lut_b} | realized fc/fs={realized:.6f} "
              f"| SQNR={sqnr:6.2f} dB")

    # -----------------------------------------------------------
    # TEST 5b-ii: The cleaner, recommended test -- fix acc_bits=16
    # (negligible frequency-rounding error at this resolution) and
    # sweep ONLY lut_bits, isolating pure LUT-truncation error.
    # -----------------------------------------------------------
    print("\n--- Test 5b-ii: LUT-only sweep at fixed acc_bits=16 "
          "(isolates truncation error) ---")
    acc_fixed = 16
    inc_fixed = round(params.GENERIC_TEST_FC_OVER_FS * (1 << acc_fixed))
    realized_fixed = inc_fixed / (1 << acc_fixed)
    cos_fl_fixed, _ = nco_float(N, realized_fixed)

    lut_sweep = [4, 6, 8, 10]
    sqnr_5bii = {}
    for lut_b in lut_sweep:
        cfx, _ = nco_fixed(N, phase_inc=inc_fixed, acc_bits=acc_fixed,
                            lut_bits=lut_b)
        cfx_n = cfx / params.DATAPATH_FULL_SCALE
        e = cfx_n - cos_fl_fixed
        sqnr = 10*np.log10(np.mean(cos_fl_fixed**2)/np.mean(e**2))
        sqnr_5bii[lut_b] = sqnr
        print(f"  lut_bits={lut_b:2d} -> SQNR={sqnr:6.2f} dB")

    monotonic = all(sqnr_5bii[lut_sweep[i]] <= sqnr_5bii[lut_sweep[i+1]] + 1e-6
                     for i in range(len(lut_sweep)-1))
    print(f"SQNR non-decreasing with lut_bits: {monotonic}")
    assert monotonic, (
        "Expected SQNR to increase monotonically with lut_bits when "
        "truncation error is properly isolated (fixed acc_bits, "
        "compared against realized frequency)"
    )
    print("PASS: this is the plot that justifies the LUT size choice.")

    # -----------------------------------------------------------
    # TEST 5c: Datapath (amplitude) precision sweep
    # -----------------------------------------------------------
    print("\n--- Test 5c: Datapath precision sweep (acc=16, lut=8 fixed) ---")
    sqnr_5c = {}
    for dp_bits in [4, 8, 12]:
        full_scale = (1 << (dp_bits - 1)) - 1
        cfx, _ = nco_fixed(N, datapath_bits=dp_bits)
        cfx_n = cfx / full_scale
        e = cfx_n - cos_fl_sys
        sqnr = 10*np.log10(np.mean(cos_fl_sys**2)/np.mean(e**2))
        sqnr_5c[dp_bits] = sqnr
        print(f"  datapath_bits={dp_bits:2d} -> SQNR={sqnr:6.2f} dB")
    assert sqnr_5c[4] < sqnr_5c[8] < sqnr_5c[12], \
        "Expected strictly increasing SQNR with datapath_bits"
    print("PASS.")

    # -----------------------------------------------------------
    # TEST 6: End-to-end noiseless constellation + self-loopback
    # -----------------------------------------------------------
    print("\n--- Test 6: Noiseless end-to-end sanity check (float domain) ---")
    n_symbols = 200
    tx_bits = gen_bits(n_symbols * 2)
    I_sym, Q_sym = map_qpsk(tx_bits)

    sps = params.SAMPLES_PER_SYMBOL
    n_samp = n_symbols * sps
    cos_c, sin_c = nco_float(n_samp, params.FC_OVER_FS)
    s_tx = modulate(I_sym, Q_sym, cos_c, sin_c, sps)

    demod_I = s_tx * cos_c
    demod_Q = -s_tx * sin_c
    I_hat = demod_I.reshape(-1, sps).mean(axis=1) * 2
    Q_hat = demod_Q.reshape(-1, sps).mean(axis=1) * 2

    err_rms = np.sqrt(np.mean((I_hat - I_sym)**2 + (Q_hat - Q_sym)**2))
    print(f"Self-loopback RMS error (expect ~0): {err_rms:.6f}")
    assert err_rms < 1e-9, "Self-loopback consistency check failed"
    print("PASS.")

    # -----------------------------------------------------------
    # TEST 7: Mixer overflow -- reproduce the bug, then verify the fix
    # -----------------------------------------------------------
    print("\n--- Test 7: Mixer headroom / overflow check ---")
    req_bits = required_mixer_bits(params.DATAPATH_BITS)
    print(f"required_mixer_bits(datapath_bits={params.DATAPATH_BITS}) "
          f"= {req_bits}  | params.MIXER_OUT_BITS = {params.MIXER_OUT_BITS}")
    assert req_bits == params.MIXER_OUT_BITS, (
        f"params.MIXER_OUT_BITS ({params.MIXER_OUT_BITS}) does not match "
        f"the value actually required ({req_bits}) for DATAPATH_BITS="
        f"{params.DATAPATH_BITS} -- update params.py"
    )

    cos_fix, sin_fix = nco_fixed(n_samp)

    # 7a: confirm the ORIGINAL bug (12-bit output) really does overflow
    try:
        modulate_fixed(I_sym, Q_sym, cos_fix, sin_fix, sps, out_bits=12)
        raise AssertionError(
            "Expected OverflowError with 12-bit mixer output, but none "
            "was raised -- overflow bug may have been masked"
        )
    except OverflowError as e:
        print(f"Confirmed: 12-bit mixer output overflows as expected -> {e}")

    # 7b: confirm the FIX (13-bit output) does not overflow
    s_fix = modulate_fixed(I_sym, Q_sym, cos_fix, sin_fix, sps,
                            out_bits=params.MIXER_OUT_BITS)
    print(f"13-bit mixer output range: [{s_fix.min()}, {s_fix.max()}] "
          f"(signed 13-bit range: [{-(1<<12)}, {(1<<12)-1}])")
    print("PASS: fixed-point mixer verified overflow-free at 13 bits.")

    # -----------------------------------------------------------
    # TEST 8: LUT hex export for Verilog $readmemh, with round-trip check
    # -----------------------------------------------------------
    print("\n--- Test 8: LUT hex export ---")
    lut_cos, lut_sin = build_nco_lut()
    export_lut_hex(lut_cos, params.DATAPATH_BITS, "lut_cos.hex")
    export_lut_hex(lut_sin, params.DATAPATH_BITS, "lut_sin.hex")

    with open("lut_cos.hex") as f:
        preview = [next(f).strip() for _ in range(4)]
    print(f"lut_cos.hex first 4 lines: {preview}")

    readback_cos = read_hex_twos_complement("lut_cos.hex", params.DATAPATH_BITS)
    readback_sin = read_hex_twos_complement("lut_sin.hex", params.DATAPATH_BITS)
    assert np.array_equal(readback_cos, lut_cos), "lut_cos.hex round-trip FAILED"
    assert np.array_equal(readback_sin, lut_sin), "lut_sin.hex round-trip FAILED"
    print("PASS: lut_cos.hex / lut_sin.hex written and verified round-trip.")
    print("These files are ready for `$readmemh(\"lut_cos.hex\", rom_cos);` "
          "in nco.v")

    # -----------------------------------------------------------
    # PLOTS (headless-safe: savefig always runs; show() is best-effort)
    # -----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0,0].scatter(I_sym, Q_sym, s=15)
    axes[0,0].set_title("Noiseless TX Constellation")
    axes[0,0].axhline(0, color='gray', lw=0.5); axes[0,0].axvline(0, color='gray', lw=0.5)
    axes[0,0].set_xlim(-1.5,1.5); axes[0,0].set_ylim(-1.5,1.5); axes[0,0].grid(True)

    axes[0,1].scatter(I_hat, Q_hat, s=15, color='orange')
    axes[0,1].set_title("Self-Loopback Recovered Constellation")
    axes[0,1].axhline(0, color='gray', lw=0.5); axes[0,1].axvline(0, color='gray', lw=0.5)
    axes[0,1].set_xlim(-1.5,1.5); axes[0,1].set_ylim(-1.5,1.5); axes[0,1].grid(True)

    spec = np.abs(np.fft.fftshift(np.fft.fft(s_tx)))
    freq_axis = np.fft.fftshift(np.fft.fftfreq(len(s_tx)))
    axes[0,2].plot(freq_axis, spec)
    axes[0,2].set_title("Spectrum of Modulated QPSK Signal"); axes[0,2].grid(True)

    labels = [f"{a}/{l}" for a,l in configs]
    x = np.arange(len(configs))
    axes[1,0].bar(x, [sqnr_5a[c] for c in configs])
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(labels)
    axes[1,0].set_title("Test 5a: system fc/fs (now correctly explained)")
    axes[1,0].set_ylabel("SQNR (dB)"); axes[1,0].grid(True, axis='y')

    axes[1,1].plot(lut_sweep, [sqnr_5bii[l] for l in lut_sweep], marker='o')
    axes[1,1].set_xlabel("lut_bits"); axes[1,1].set_ylabel("SQNR (dB)")
    axes[1,1].set_title("Test 5b-ii: LUT truncation (isolated, vs realized freq)")
    axes[1,1].grid(True)

    dp_list = sorted(sqnr_5c.keys())
    axes[1,2].plot(dp_list, [sqnr_5c[d] for d in dp_list], marker='o', color='green')
    axes[1,2].set_xlabel("datapath_bits"); axes[1,2].set_ylabel("SQNR (dB)")
    axes[1,2].set_title("Test 5c: amplitude-quantization sweep")
    axes[1,2].grid(True)

    plt.tight_layout()
    plt.savefig("part_a_verification_plots.png", dpi=150)
    print("\nPlots saved to part_a_verification_plots.png")
    try:
        plt.show()
    except Exception as e:
        print(f"(plt.show() skipped -- no display available: {e})")

    print("\n" + "=" * 70)
    print("All Part A self-tests completed and PASSED (every check above")
    print("is backed by an assert -- reaching this line means nothing failed).")
    print("=" * 70)