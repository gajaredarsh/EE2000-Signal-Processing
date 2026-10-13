"""
rx_backend_eval.py
Part D of the QPSK FPGA project golden model: the RECEIVER BACK END and evaluation.

Owns:
    - Integrate-and-dump symbol sampling   -> integrate_dump()
    - Sign decision / demapper             -> decide()
    - BER counter                          -> count_errors(), ber()
    - Sweep bookkeeping and statistics     -> bits_needed(), ber_sweep(), z_score()
    - Plots                                -> plot_ber(), plot_constellations()
    - Hex export of golden vectors         -> fits_signed(), write_vectors()

Author: Darsh (Part D)
AI-assistance note: first draft written with Claude (Sonnet 5.5, Anthropic) as allowed by the
    EE2000 AI-usage policy. The author must be able to explain and defend all of it in the viva.

---------------------------------------------------------------------------------------
Where this block sits (signal flow)
---------------------------------------------------------------------------------------
    Part A: bits -> (I,Q) -> s[n] = I*cos[n] - Q*sin[n]
    Part B: r[n] = s[n] + noise
    Part C: r[n] -> mixer -> (FIR) -> yi, yq                  (baseband level +/-0.5)
    Part D: sum SPS samples per symbol -> sign -> bits -> count errors      (THIS FILE)

Interface contract:
    integrate_dump(yi, yq, timing_offset=0) -> (I_hat, Q_hat)   one value per symbol
    decide(I_hat, Q_hat)                    -> bits             flat, [b0, b1, b0, b1, ...]
    count_errors(tx_bits, rx_bits)          -> (errors, total)
    ber(tx_bits, rx_bits)                   -> float

Conventions (all come from params.py and rx_frontend.py):
 1. Mapping is I = 1-2*b0, Q = 1-2*b1, so a NEGATIVE decision value means bit 1. Ties go to 0,
    the same rule as tx_nco.demap_qpsk_reference().
 2. The baseband level is +/-0.5, so a noiseless symbol sums to +/-SPS/2 (NOISELESS_LEVEL).
    Divide by NOISELESS_LEVEL to draw the constellation at +/-1. Decisions only use the sign.
 3. Causal FIR output: symbol k starts at sample SPS*k + D, so pass timing_offset = D
    (C.group_delay_samples(h)). With rx_frontend(..., aligned=True) pass timing_offset = 0.
 4. The first and last symbol see filter edge effects, so count_errors() skips one at each end.
 5. Integer inputs are summed in int64, so the accumulator is exact. In hardware it needs
    ACC_BITS = RX_IN_BITS + log2(SPS) = 16 signed bits (16 samples of at most 12 bits each).
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import params
import channel
import fixedpoint
import rx_frontend as C

SPS = params.SAMPLES_PER_SYMBOL
NOISELESS_LEVEL = SPS / 2.0                                  # |I_hat| for a clean symbol, float chain
ACC_BITS = C.RX_IN_BITS + int(np.ceil(np.log2(SPS)))         # worst-case integrate-and-dump width


# =======================================================================================
# 1. Symbol sampling and decision
# =======================================================================================

def integrate_dump(yi, yq, timing_offset=0, sps=SPS):
    """
    Sum every block of `sps` samples (after skipping `timing_offset` samples) into one value.
    This is the matched filter for a rectangular pulse: a boxcar of length sps, read once per
    symbol. Returns two arrays with one entry per COMPLETE symbol.
    """
    yi = np.asarray(yi)[timing_offset:]
    yq = np.asarray(yq)[timing_offset:]
    n_sym = min(len(yi), len(yq)) // sps
    if n_sym == 0:
        raise ValueError("not enough samples for a single symbol")
    if np.issubdtype(yi.dtype, np.integer):
        yi = yi.astype(np.int64)
    if np.issubdtype(yq.dtype, np.integer):
        yq = yq.astype(np.int64)
    I_hat = yi[:n_sym * sps].reshape(n_sym, sps).sum(axis=1)
    Q_hat = yq[:n_sym * sps].reshape(n_sym, sps).sum(axis=1)
    return I_hat, Q_hat


def decide(I_hat, Q_hat):
    """Sign -> bit (negative = 1). Returns a flat uint8 array [b0, b1, b0, b1, ...]."""
    I_hat = np.asarray(I_hat)
    Q_hat = np.asarray(Q_hat)
    bits = np.empty(2 * len(I_hat), dtype=np.uint8)
    bits[0::2] = I_hat < 0
    bits[1::2] = Q_hat < 0
    return bits


# =======================================================================================
# 2. BER counter
# =======================================================================================

def count_errors(tx_bits, rx_bits, skip_first=1, skip_last=1):
    """
    Compare bit streams symbol by symbol and return (errors, bits_compared).
    The streams may differ in length (a causal filter cuts off the last symbol); only the
    common complete symbols are compared. skip_first / skip_last drop edge symbols.
    """
    tx_bits = np.asarray(tx_bits)
    rx_bits = np.asarray(rx_bits)
    n_sym = min(len(tx_bits), len(rx_bits)) // 2
    lo, hi = skip_first, n_sym - skip_last
    if hi <= lo:
        raise ValueError("no symbols left to compare after skipping the edges")
    a = tx_bits[2 * lo:2 * hi]
    b = rx_bits[2 * lo:2 * hi]
    return int(np.count_nonzero(a != b)), int(a.size)


def ber(tx_bits, rx_bits, skip_first=1, skip_last=1):
    errors, total = count_errors(tx_bits, rx_bits, skip_first, skip_last)
    return errors / total


# =======================================================================================
# 3. Sweep bookkeeping
# =======================================================================================

def bits_needed(expected_ber, min_errors=100, max_bits=1_500_000):
    """Bits to simulate so that about `min_errors` errors are expected (~10% accuracy)."""
    n = int(np.ceil(min_errors / max(expected_ber, 1e-12)))
    return min(n, max_bits)


def z_score(errors, bits, expected_ber):
    """How many standard deviations the measured error count is from the expected one."""
    mean = bits * expected_ber
    sd = np.sqrt(bits * expected_ber * (1.0 - expected_ber))
    return (errors - mean) / sd


def ber_sweep(run_chunk, ebn0_list, names, min_errors=100, max_bits=1_500_000,
              chunk_sym=50_000, seed0=1000, size_ber=channel.theory_ber):
    """
    Run the chain at every Eb/N0 in ebn0_list.
        run_chunk(ebn0_db, n_sym, seed) -> {name: (errors, bits)}   (supplied by main.py)
    Each point is built from chunks of chunk_sym symbols so memory stays small, and the number
    of chunks is chosen so the LOWEST expected BER (size_ber) still sees ~min_errors errors.
    Returns {name: {"ebn0": [...], "errors": [...], "bits": [...], "ber": [...]}}.
    """
    out = {n: {"ebn0": [], "errors": [], "bits": [], "ber": []} for n in names}
    for idx, e in enumerate(ebn0_list):
        target = bits_needed(size_ber(e), min_errors, max_bits)
        n_chunks = max(1, int(np.ceil(target / (2 * chunk_sym))))
        errs = {n: 0 for n in names}
        tot = {n: 0 for n in names}
        for k in range(n_chunks):
            res = run_chunk(e, chunk_sym, seed0 + 1000 * idx + k)
            for n in names:
                errs[n] += res[n][0]
                tot[n] += res[n][1]
        for n in names:
            out[n]["ebn0"].append(e)
            out[n]["errors"].append(errs[n])
            out[n]["bits"].append(tot[n])
            out[n]["ber"].append(errs[n] / tot[n])
    return out


# =======================================================================================
# 4. Plots
# =======================================================================================

def plot_ber(sim, lines, path):
    """
    sim   : {label: (ebn0_array, ber_array)}   drawn as markers
    lines : {label: (ebn0_array, ber_array)}   drawn as curves
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (x, y) in lines.items():
        ax.semilogy(x, y, "-", label=label)
    for label, (x, y) in sim.items():
        ax.semilogy(x, y, "o", mfc="none", ms=8, label=label)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_ylim(1e-5, 1)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title("QPSK BER vs Eb/N0")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_constellations(panels, path, scale=NOISELESS_LEVEL):
    """panels: list of (title, I_hat, Q_hat). Decision values are divided by `scale`."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), squeeze=False)
    for ax, (title, I_hat, Q_hat) in zip(axes[0], panels):
        ax.scatter(np.asarray(I_hat) / scale, np.asarray(Q_hat) / scale, s=3, alpha=0.4)
        ax.plot([-1, -1, 1, 1], [-1, 1, -1, 1], "r+", ms=14, mew=2)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =======================================================================================
# 5. Hex export for the Verilog testbench
# =======================================================================================

def fits_signed(x, bits):
    x = np.asarray(x)
    return bool(x.min() >= -(1 << (bits - 1)) and x.max() <= (1 << (bits - 1)) - 1)


def write_vectors(out_dir, vectors):
    """vectors: {filename: (codes, bits)}. Uses Part B's write_hex (two's complement, one per line)."""
    os.makedirs(out_dir, exist_ok=True)
    for name, (codes, bits) in vectors.items():
        fixedpoint.write_hex(os.path.join(out_dir, name), codes, bits)


# =======================================================================================
# 6. Self-test (this file alone, no other part needed except the shared helpers)
# =======================================================================================

if __name__ == "__main__":
    import tx_nco

    print("=" * 74)
    print("Part D self-test: rx_backend_eval.py")
    print("=" * 74)

    rng = np.random.default_rng(0)

    print("\n--- Test 1: integrate_dump sums blocks and respects timing_offset ---")
    x = np.arange(3 * SPS, dtype=float)
    I_hat, Q_hat = integrate_dump(x, -x)
    expected = [x[k * SPS:(k + 1) * SPS].sum() for k in range(3)]
    assert np.allclose(I_hat, expected) and np.allclose(Q_hat, -np.array(expected))
    I_off, _ = integrate_dump(x, x, timing_offset=5)
    assert len(I_off) == 2 and np.isclose(I_off[0], x[5:5 + SPS].sum())
    print("PASS: block sums and offset are correct; partial trailing block is dropped.")

    print("\n--- Test 2: integer inputs are summed exactly in int64 ---")
    big = np.full(SPS * 4, 2047, dtype=np.int16)
    I_int, _ = integrate_dump(big, big)
    assert I_int.dtype == np.int64 and np.all(I_int == 2047 * SPS)
    assert fits_signed(I_int, ACC_BITS) and not fits_signed(I_int, ACC_BITS - 1)
    print(f"PASS: worst-case sum {2047 * SPS} fits {ACC_BITS} signed bits and not {ACC_BITS - 1}.")

    print("\n--- Test 3: decide() inverts Part A's mapper (all four symbols + tie) ---")
    pairs = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
    I_s, Q_s = tx_nco.map_qpsk(pairs)
    assert np.array_equal(decide(I_s, Q_s), pairs)
    assert np.array_equal(decide(I_s, Q_s), tx_nco.demap_qpsk_reference(I_s, Q_s))
    assert np.array_equal(decide([0.0], [0.0]), [0, 0]), "tie must decide 0"
    print("PASS: matches the locked table and Part A's reference demapper.")

    print("\n--- Test 4: count_errors skips edges and handles unequal lengths ---")
    tx = rng.integers(0, 2, 40).astype(np.uint8)
    rx = tx.copy()
    rx[0] ^= 1                                  # error in the first symbol (skipped)
    rx[-1] ^= 1                                 # error in the last symbol (skipped)
    assert count_errors(tx, rx) == (0, 36)
    rx[10] ^= 1
    assert count_errors(tx, rx) == (1, 36)
    assert count_errors(tx, rx[:-2]) == (1, 34)  # one symbol shorter (causal cut-off)
    assert abs(ber(tx, rx) - 1 / 36) < 1e-12
    print("PASS.")

    print("\n--- Test 5: bits_needed and z_score ---")
    assert bits_needed(1e-2) == 10_000 and bits_needed(1e-9) == 1_500_000
    assert abs(z_score(100, 10_000, 1e-2)) < 1e-9
    assert z_score(130, 10_000, 1e-2) > 2.5
    print("PASS.")

    print("\n--- Test 6: noiseless float chain, decision values are +/-SPS/2 ---")
    bits = rng.integers(0, 2, 2000).astype(np.uint8)
    I, Q = tx_nco.map_qpsk(bits)
    cos, sin = tx_nco.nco_float(len(I) * SPS)
    s = tx_nco.modulate(I, Q, cos, sin, SPS)
    xi, xq = C.downconvert(s, cos, sin)
    I_hat, Q_hat = integrate_dump(xi, xq)
    assert np.allclose(I_hat / NOISELESS_LEVEL, I) and np.allclose(Q_hat / NOISELESS_LEVEL, Q)
    assert count_errors(bits, decide(I_hat, Q_hat)) == (0, 2 * (len(I) - 2))
    print("PASS: integrate-and-dump alone recovers the symbols exactly, 0 errors.")

    print("\n" + "=" * 74)
    print("All Part D self-tests completed and PASSED (every check above is an assert).")
    print("=" * 74)
