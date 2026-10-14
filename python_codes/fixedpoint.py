# fixedpoint.py

import numpy as np

def quantize(x, bits, full_scale=1.0):
    levels = 2**(bits - 1)
    codes = np.round(np.array(x) * levels / full_scale)
    codes = np.clip(codes, -levels, levels - 1)
    return codes * full_scale / levels

def to_int(x, bits, full_scale=1.0):
    levels = 2**(bits - 1)
    codes = np.round(np.array(x) * levels / full_scale)
    codes = np.clip(codes, -levels, levels - 1)
    return codes.astype(int)

def clip_fraction(x, bits, full_scale=1.0):
    levels = 2**(bits - 1)
    codes = np.round(np.array(x) * levels / full_scale)
    clipped = (codes < -levels) | (codes > levels - 1)
    return np.mean(clipped)

def to_hex(codes, bits):
    width = (bits + 3) // 4
    mask = (1 << bits) - 1
    out = []
    for c in codes:
        out.append(format(int(c) & mask, '0%dx' % width))
    return out

def write_hex(filename, codes, bits):
    lines = to_hex(codes, bits)
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')

if __name__ == "__main__":
    x = np.linspace(-1, 1, 10)
    q = quantize(x, 8)
    print("input: ", x)
    print("quantized:", q)

    codes = to_int(x, 8)
    print("codes:", codes)

    big = np.array([-5, 5, 0.5])
    print("clip fraction test:", clip_fraction(big, 8, full_scale=1.0))

    write_hex("test.hex", codes, 8)
    print("wrote test.hex")
