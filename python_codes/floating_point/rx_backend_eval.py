import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
import matplotlib.pyplot as plt
import params
import fixedpoint
import channel
import rx_frontend as C
import tx_nco

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

