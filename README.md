## Overview
This repository contains code for performing generalized quantum measurements (POVMs) on multi-qubit systems. The implementations include Naimark's dilation, binary search, and a new hybrid approach which combines both.

## Key Features
- **Naimark's Dilation**: Implements POVMs by coupling the measured system to an ancillary system and performing a single projective measurement in the extended Hilbert space.
- **Binary Search Method**: Uses a single auxiliary qubit for multi-qubit POVMs through conditional measurements. We implement the binary tree protocol devised by [Andersson and Oi](https://arxiv.org/abs/0712.2665).
- **Hybrid Approach (Naimark-Terminated Binary Tree)**: Combines Naimark's dilation and binary search for more efficient circuits.
- **Conditional Readout Error Mitigation (CREM)**: Combats readout error propagation in dynamic circuits.


![two_qubit_schematic](https://github.com/ibm-q-collaboration/qamp-generalized-measurements/assets/63845272/90c7ae76-05cc-4848-b9b1-8ec31d315f63)
