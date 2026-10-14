# QPSK Modulator and Demodulator with Simulated AWGN Channel

This repository contains the implementation of a Quadrature Phase Shift Keying(QPSK) system along with a AWGN channel on Spartan-7 Boolean board, along with its simulation using Python.
The project models the complete digital communication chain, from bit generation and modulation to channel simulation, coherent demodulation, bit recovery, and Bit Error Rate (BER) analysis.

## Project Context

This project was developed as part of coursework at the Indian Institute of Technology Hyderabad (IITH) to explore the design and implementation of a digital QPSK communication system.

**Course:** EE2000 - Signal Processing
**Project Type:**
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

```mermaid
flowchart TB
    subgraph TX["TRANSMITTER"]
        A["Input Bit Stream"]
        B["QPSK Modulator"]
        C["Digital Carrier Generation"]
        A --> B --> C
    end

    subgraph CH["CHANNEL"]
        D["Channel Model"]
        E["AWGN / LFSR Noise"]
        D --- E
    end

    subgraph RX["RECEIVER"]
        F["I/Q Downconversion"]
        G["FIR Low-Pass Filters"]
        H["Symbol Integration"]
        I["Bit Decisions"]
        F --> G --> H --> I
    end

    subgraph EV["EVALUATION"]
        J["Recovered Bit Stream"]
        K["BER Analysis"]
        L["Performance Evaluation"]
        J --> K --> L
    end

    C --> D
    D --> F
    I --> J

    M["Fixed-Point Processing & Golden-Vector Generation"]

    M -.-> C
    M -.-> F
    M -.-> G
    M -.-> I

    classDef completed fill:#dcefe2,stroke:#39845b,color:#173d28,stroke-width:1.5px
    classDef planned fill:#fff0d5,stroke:#c58b32,color:#66400c,stroke-width:1.5px
    classDef evaluation fill:#dce9fa,stroke:#4778b8,color:#193b65,stroke-width:1.5px

    class A,B,C,D,E,F,G,H,I,J,K,L completed
    class M planned
    ```
## Project Status

As of yet the simulation of the project has been completed in Python and are placed under the folder python_codes along with the steps to run it. 
Relevant Literature reviewed for the project has been placed under the folder Literature.
The repository also contains Preliminary Report and Mid-Term Report required for the coursework evaluation.
