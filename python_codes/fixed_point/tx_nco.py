import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import params


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
    b0 = (np.asarray(I) < 0).astype(np.uint8)
    b1 = (np.asarray(Q) < 0).astype(np.uint8)
    bits = np.empty(2 * len(I), dtype=np.uint8)
    bits[0::2] = b0
    bits[1::2] = b1
    return bits

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

if __name__ == "__main__":
    lut_cos, lut_sin = build_nco_lut()
    export_lut_hex(lut_cos, params.DATAPATH_BITS, "lut_cos.hex")
    export_lut_hex(lut_sin, params.DATAPATH_BITS, "lut_sin.hex")
    print("Wrote lut_cos.hex and lut_sin.hex "
          f"({params.NCO_LUT_SIZE} entries, {params.DATAPATH_BITS}-bit two's complement)")
