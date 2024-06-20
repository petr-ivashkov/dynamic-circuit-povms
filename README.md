## Overview
This repository contains code for performing generalized quantum measurements (POVMs) on multi-qubit systems. The implementations include Naimark's dilation, binary search, and a new hybrid approach that combines both. 

## Key Features
- **Naimark's Dilation**: Implements POVMs by coupling the measured system to an ancillary system and performing a single projective measurement in the extended Hilbert space.
- **Binary Search Method**: Uses a single auxiliary qubit for multi-qubit POVMs through conditional measurements. We implement the binary tree protocol devised by [Andersson and Oi](https://arxiv.org/abs/0712.2665).
- **Hybrid Approach (Naimark-Terminated Binary Tree)**: Combines Naimark's dilation and binary search for more efficient circuits.
- **Conditional Readout Error Mitigation (CREM)**: Combats readout error propagation in dynamic circuits.

## Installation

1. **Clone the repository**:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the required dependencies** using `requirements.txt`:
    ```sh
    pip install -r requirements.txt
    ```

3. **Install the `src` package** by navigating in the project directory and running:
    ```sh
    pip install -e .
    ```

## Jupyter Notebook

The provided Jupyter notebook demonstrates the following:

1. **Loading SIC-POVMs**: Load SIC-POVMs in 2D and 4D spaces.
2. **Generating Circuits**: Construct quantum circuits for Naimark, Binary Tree, and Hybrid Tree POVMs.
3. **Transpiling Circuits**: Transpile the generated circuits.
4. **Noiseless Simulation**: Run noiseless simulations and compare the results with theoretical probabilities.
5. **Noisy Simulation**: Run noisy simulations and compare the results with theoretical probabilities.
6. **Readout Error Mitigation**: Apply conditional readout error mitigation.

This project is licensed under the MIT License. 
