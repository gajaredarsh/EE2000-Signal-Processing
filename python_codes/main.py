"""
main.py
Part D of the QPSK FPGA project golden model: the END-TO-END RUNNER.

Wires Part A (tx_nco) + Part B (channel, fixedpoint) + Part C (rx_frontend) + Part D
(rx_backend_eval) into one pipeline and runs the checks that gate the Verilog work:

    1. noiseless chain gives exactly 0 errors (float and fixed point)
    2. causal FIR + timing_offset = D gives the same decisions as aligned FIR + offset 0
    3. BER sweep, with and without the FIR, against theory and Part C's exact curve
    4. constellation plots
    5. fixed-point chain gives the same BER as the float chain
    6. hardware-style LFSR noise gives the predicted BER (short run, see note below)
    7. golden hex vectors for the Part D Verilog testbench

Run:   python main.py            (full run, a few minutes)
       python main.py --quick    (smaller sweep, for a fast check)

Author: Darsh (Part D)
AI-assistance note: first draft written with Claude (Sonnet 5.5, Anthropic) as allowed by the
    EE2000 AI-usage policy. The author must be able to explain and defend all of it in the viva.

Notes:
  * Sweeps use numpy noise. channel.py's LFSR noise repeats exactly every 65535 samples
    (about 4096 symbols at 16 samples/symbol), so it is only used for short runs.
  * Sweep bits are numpy random, not the PRBS-7 (period 127 bits), so the statistics are not
    tied to one short repeating pattern. The PRBS is used for the hardware golden vectors.
"""

import os
import sys

import numpy as np

import params
import channel
import fixedpoint
import tx_nco
import rx_frontend as C
import rx_backend_eval as D

SPS = params.SAMPLES_PER_SYMBOL
QUICK = "--quick" in sys.argv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "d_outputs")
GOLDEN_DIR = os.path.join(OUT_DIR, "golden")

H = C.design_lpf()                              # float taps, order 24
H_INT = C.quantize_coeffs(H, C.COEF_BITS)       # 12-bit taps, DC gain exactly 1
D_FIR = C.group_delay_samples(H)                # 12 samples


# =======================================================================================
# Chain building blocks
# =======================================================================================

def tx_float(bits):
    I, Q = tx_nco.map_qpsk(bits)
    cos, sin = tx_nco.nco_float(len(I) * SPS)
    s = tx_nco.modulate(I, Q, cos, sin, SPS)
    return I, Q, s, cos, sin


def tx_fixed(bits):
    """Integer TX. Phase is back to 0 at every symbol (2 whole carrier cycles), so one
    symbol of NCO output tiled n_sym times equals running the NCO for the whole run."""
    I, Q = tx_nco.map_qpsk(bits)
    n_sym = len(I)
    c1, s1 = tx_nco.nco_fixed(SPS)
    cos_int = np.tile(c1, n_sym)
    sin_int = np.tile(s1, n_sym)
    s_fix = tx_nco.modulate_fixed(I, Q, cos_int, sin_int, SPS)
    return I, Q, s_fix, cos_int, sin_int


def rx_float(r, cos, sin, h, aligned=True):
    """h=None means no FIR: mixer output goes straight to integrate-and-dump."""
    if h is None:
        return C.downconvert(r, cos, sin)
    return C.rx_frontend(r, cos, sin, h, aligned=aligned)


def rx_fixed(r_int, cos_int, sin_int, h_int, aligned=True):
    if h_int is None:
        return C.downconvert_fixed(r_int, cos_int, sin_int)
    return C.rx_frontend_fixed(r_int, cos_int, sin_int, h_int, aligned=aligned)


def decode(yi, yq, timing_offset=0):
    I_hat, Q_hat = D.integrate_dump(yi, yq, timing_offset)
    return I_hat, Q_hat, D.decide(I_hat, Q_hat)


def simulate_float(ebn0_db, n_sym, seed, noise_mode="numpy"):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, 2 * n_sym).astype(np.uint8)
    I, Q, s, cos, sin = tx_float(bits)
    r = channel.awgn(s, ebn0_db, mode=noise_mode, seed=seed + 1)
    return bits, r, cos, sin


def run_chunk(ebn0_db, n_sym, seed):
    """One chunk of the sweep: same noisy r decoded without and with the FIR."""
    bits, r, cos, sin = simulate_float(ebn0_db, n_sym, seed)
    out = {}
    for name, h in (("no_fir", None), ("fir", H)):
        yi, yq = rx_float(r, cos, sin, h, aligned=True)
        out[name] = D.count_errors(bits, decode(yi, yq)[2])
    return out


# =======================================================================================
# 1. Noiseless sanity (gate: must be exactly zero errors)
# =======================================================================================

def check_noiseless():
    print("\n--- Check 1: noiseless chain must give exactly 0 errors ---")
    bits = np.random.default_rng(1).integers(0, 2, 4000).astype(np.uint8)

    I, Q, s, cos, sin = tx_float(bits)
    for name, h, aligned, off in (("float, no FIR", None, True, 0),
                                  ("float, FIR aligned", H, True, 0),
                                  ("float, FIR causal", H, False, D_FIR)):
        yi, yq = rx_float(s, cos, sin, h, aligned)
        errs, total = D.count_errors(bits, decode(yi, yq, off)[2])
        print(f"  {name:20s}: {errs} errors / {total} bits")
        assert errs == 0, f"{name}: chain is broken, do not trust any BER number"

    I, Q, s_fix, cos_i, sin_i = tx_fixed(bits)
    r_int = C.quantize_rx(s_fix / params.DATAPATH_FULL_SCALE)
    for name, h_int in (("fixed, no FIR", None), ("fixed, FIR", H_INT)):
        yi, yq = rx_fixed(r_int, cos_i, sin_i, h_int, aligned=True)
        I_hat, Q_hat, rx_bits = decode(yi, yq)
        errs, total = D.count_errors(bits, rx_bits)
        print(f"  {name:20s}: {errs} errors / {total} bits")
        assert errs == 0, f"{name}: fixed-point chain is broken"
        assert D.fits_signed(I_hat, D.ACC_BITS) and D.fits_signed(Q_hat, D.ACC_BITS)
    print("PASS")


# =======================================================================================
# 2. Causal vs aligned timing
# =======================================================================================

def check_timing():
    print("\n--- Check 2: causal FIR + offset D equals aligned FIR + offset 0 ---")
    bits, r, cos, sin = simulate_float(6.0, 500, seed=9)
    yi_c, yq_c = rx_float(r, cos, sin, H, aligned=False)
    yi_a, yq_a = rx_float(r, cos, sin, H, aligned=True)
    Ic, Qc, _ = decode(yi_c, yq_c, timing_offset=D_FIR)
    Ia, Qa, _ = decode(yi_a, yq_a, timing_offset=0)
    n = len(Ic)
    assert np.allclose(Ic, Ia[:n]) and np.allclose(Qc, Qa[:n])
    print(f"  {n} of {len(Ia)} symbols compared (causal mode cuts off the last one), identical.")
    wrong = decode(yi_c, yq_c, timing_offset=0)[0]
    assert not np.allclose(wrong[:n], Ia[:n])
    print("  and timing_offset = 0 on the causal stream does NOT match, so the offset matters.")
    print("PASS")


# =======================================================================================
# 3. BER sweep
# =======================================================================================

def sweep():
    print("\n--- Check 3: BER vs Eb/N0, without and with the FIR ---")
    ebn0_list = [0, 2, 4, 6, 8]
    max_bits = 200_000 if QUICK else 1_500_000
    names = ["no_fir", "fir"]
    res = D.ber_sweep(run_chunk, ebn0_list, names, min_errors=100, max_bits=max_bits)

    theory = [channel.theory_ber(e) for e in ebn0_list]
    exact_no = [C.ber_semi_analytic(None, e) for e in ebn0_list]
    exact_fir = [C.ber_semi_analytic(H, e) for e in ebn0_list]
    print(f"{'Eb/N0':>6s} {'bits':>9s} | {'no-FIR':>9s} {'theory':>9s} {'z':>5s} | "
          f"{'FIR':>9s} {'exact':>9s} {'z':>5s} | {'FIR cost':>8s}")
    for k, e in enumerate(ebn0_list):
        b0, b1 = res["no_fir"], res["fir"]
        z0 = D.z_score(b0["errors"][k], b0["bits"][k], exact_no[k])
        z1 = D.z_score(b1["errors"][k], b1["bits"][k], exact_fir[k])
        cost = C.ber_penalty_db(H, e)
        print(f"{e:6.1f} {b0['bits'][k]:9d} | {b0['ber'][k]:9.3e} {theory[k]:9.3e} {z0:5.1f} | "
              f"{b1['ber'][k]:9.3e} {exact_fir[k]:9.3e} {z1:5.1f} | {cost:7.3f}dB")
        if not QUICK:
            assert abs(z0) < 4 and abs(z1) < 4, "simulation disagrees with the exact prediction"

    fine = np.arange(0, 8.01, 0.25)
    D.plot_ber(
        sim={"simulated, no FIR": (ebn0_list, res["no_fir"]["ber"]),
             "simulated, FIR": (ebn0_list, res["fir"]["ber"])},
        lines={"theory Q(sqrt(2Eb/N0))": (fine, [channel.theory_ber(e) for e in fine]),
               "exact, FIR (Part C)": (fine, [C.ber_semi_analytic(H, e) for e in fine])},
        path=os.path.join(OUT_DIR, "ber_curve.png"))
    print("PASS: both simulated curves sit on their predictions."
          if not QUICK else "(quick mode: statistics not asserted)")
    print(f"wrote {os.path.join(OUT_DIR, 'ber_curve.png')}")


# =======================================================================================
# 4. Constellations
# =======================================================================================

def constellations():
    print("\n--- Check 4: constellation plots (FIR, aligned) ---")
    panels = []
    for e in (0.0, 6.0, 12.0):
        bits, r, cos, sin = simulate_float(e, 3000, seed=21)
        yi, yq = rx_float(r, cos, sin, H, aligned=True)
        I_hat, Q_hat, _ = decode(yi, yq)
        panels.append((f"Eb/N0 = {e:.0f} dB", I_hat[1:-1], Q_hat[1:-1]))
    D.plot_constellations(panels, os.path.join(OUT_DIR, "constellations.png"))
    print(f"wrote {os.path.join(OUT_DIR, 'constellations.png')}")


# =======================================================================================
# 5. Fixed point vs float
# =======================================================================================

def check_fixed_vs_float():
    print("\n--- Check 5: fixed-point chain vs float chain at 4 dB ---")
    n_sym = 40_000 if QUICK else 100_000
    bits = np.random.default_rng(5).integers(0, 2, 2 * n_sym).astype(np.uint8)

    I, Q, s, cos, sin = tx_float(bits)
    r_fl = channel.awgn(s, 4.0, mode="numpy", seed=11)
    yi, yq = rx_float(r_fl, cos, sin, H, aligned=True)
    ber_fl = D.ber(bits, decode(yi, yq)[2])

    I, Q, s_fix, cos_i, sin_i = tx_fixed(bits)
    r_fx = channel.awgn(s_fix / params.DATAPATH_FULL_SCALE, 4.0, mode="numpy", seed=11)
    clip = fixedpoint.clip_fraction(r_fx, C.RX_IN_BITS, C.RX_FULL_SCALE)
    yi, yq = rx_fixed(C.quantize_rx(r_fx), cos_i, sin_i, H_INT, aligned=True)
    I_hat, Q_hat, rx_bits = decode(yi, yq)
    ber_fx = D.ber(bits, rx_bits)

    print(f"  input clipping fraction {clip:.2e} | max |accumulator| "
          f"{int(max(np.abs(I_hat).max(), np.abs(Q_hat).max()))} (limit {(1 << (D.ACC_BITS - 1)) - 1})")
    print(f"  BER float {ber_fl:.5f} | BER fixed {ber_fx:.5f}")
    assert D.fits_signed(I_hat, D.ACC_BITS) and D.fits_signed(Q_hat, D.ACC_BITS)
    assert abs(ber_fx - ber_fl) / ber_fl < 0.05, "fixed-point chain changes the BER"
    print("PASS: integer chain matches the float chain, accumulator never overflows.")


# =======================================================================================
# 6. LFSR (hardware-style) noise, short run
# =======================================================================================

def check_lfsr_noise():
    print("\n--- Check 6: hardware-style LFSR noise, 4000 symbols at 2 dB ---")
    bits = tx_nco.gen_bits(2 * 4000)
    I, Q, s, cos, sin = tx_float(bits)
    r = channel.awgn(s, 2.0, mode="lfsr", seed=3)
    yi, yq = rx_float(r, cos, sin, H, aligned=True)
    measured = D.ber(bits, decode(yi, yq)[2])
    predicted = C.ber_semi_analytic(H, 2.0)
    print(f"  BER measured {measured:.4f} | exact prediction {predicted:.4f}")
    assert abs(measured - predicted) / predicted < 0.20
    print("PASS")


# =======================================================================================
# 7. Golden vectors for the Verilog testbench
# =======================================================================================

def export_golden(n_sym=128, ebn0_db=6.0):
    print("\n--- Check 7: golden vectors for the integrate-and-dump + decision RTL ---")
    bits = tx_nco.gen_bits(2 * n_sym)                       # PRBS-7, same as the RTL source
    I, Q, s_fix, cos_i, sin_i = tx_fixed(bits)
    r = channel.awgn(s_fix / params.DATAPATH_FULL_SCALE, ebn0_db, mode="lfsr", seed=3)
    yi, yq = rx_fixed(C.quantize_rx(r), cos_i, sin_i, H_INT, aligned=False)   # causal, like the RTL
    acc_i, acc_q, rx_bits = decode(yi, yq, timing_offset=D_FIR)
    assert D.fits_signed(acc_i, D.ACC_BITS) and D.fits_signed(acc_q, D.ACC_BITS)
    n = len(acc_i)
    errs, total = D.count_errors(bits, rx_bits)

    D.write_vectors(GOLDEN_DIR, {
        "d_yi.hex": (yi, C.RX_IN_BITS),          # input stream, I branch   (12 bit)
        "d_yq.hex": (yq, C.RX_IN_BITS),          # input stream, Q branch   (12 bit)
        "d_acc_i.hex": (acc_i, D.ACC_BITS),      # expected accumulator, I  (16 bit)
        "d_acc_q.hex": (acc_q, D.ACC_BITS),      # expected accumulator, Q  (16 bit)
        "d_bits_tx.hex": (bits[:2 * n], 1),      # transmitted bits         (1 bit)
        "d_bits_rx.hex": (rx_bits, 1),           # expected decisions       (1 bit)
    })
    with open(os.path.join(GOLDEN_DIR, "info.txt"), "w") as f:
        f.write(f"samples per symbol : {SPS}\n")
        f.write(f"timing offset      : {D_FIR} samples (causal FIR group delay)\n")
        f.write(f"input width        : {C.RX_IN_BITS} bits signed\n")
        f.write(f"accumulator width  : {D.ACC_BITS} bits signed\n")
        f.write(f"symbols exported   : {n}\n")
        f.write(f"channel            : lfsr noise, Eb/N0 = {ebn0_db} dB\n")
        f.write(f"errors (edges skipped): {errs} / {total}\n")
    print(f"  wrote 6 hex files + info.txt to {GOLDEN_DIR}")
    print(f"  {n} symbols, offset {D_FIR}, accumulators {D.ACC_BITS} bit, {errs} errors / {total} bits")
    print("PASS")


# =======================================================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 74)
    print("Part D end-to-end run" + ("  (quick mode)" if QUICK else ""))
    print("=" * 74)

    check_noiseless()
    check_timing()
    sweep()
    constellations()
    check_fixed_vs_float()
    check_lfsr_noise()
    export_golden()

    print("\n" + "=" * 74)
    print("All Part D checks completed and PASSED (every check above is an assert).")
    print("=" * 74)
