# QPSK Modulator and Demodulator with Simulated AWGN Channel

This repository contains the implementation of a Quadrature Phase Shift Keying(QPSK) system along with a AWGN channel on Spartan-7 Boolean board, along with its simulation using Python.
The project models the complete digital communication chain, from bit generation and modulation to channel simulation, coherent demodulation, bit recovery, and Bit Error Rate (BER) analysis.

## Features

- QPSK modulation and coherent demodulation
- In-phase (I) and Quadrature (Q) signal processing
- Digital carrier generation using a Numerically Controlled Oscillator (NCO)
- AWGN channel simulation
- LFSR-based noise generation
- FIR low-pass filtering using a Hamming window
- Symbol integration and bit decisions
- Bit Error Rate (BER) calculation
- Theoretical and simulated BER comparison
- Fixed-point quantization and saturation
- Hexadecimal vector export for hardware testing
- Constellation and performance visualizations
- Automated verification of the communication pipeline

## System Overview

The communication system consists of the following stages:

```text
                TRANSMITTER
                     |
              Input Bit Stream
                     |
               QPSK Modulator
                     |
              Digital Carriers
                     |
                     v
              Channel Model
             (AWGN / LFSR Noise)
                     |
                     v
                RECEIVER
                     |
              I/Q Downconversion
                     |
                FIR Filtering
                     |
             Symbol Integration
                     |
               Bit Decisions
                     |
                     v
              Recovered Bits
                     |
                     v
              BER Evaluation
