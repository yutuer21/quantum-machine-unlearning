#!/usr/bin/env python3
"""
Example script demonstrating how to run the machine unlearning experiments.
"""

import numpy as np
from models import *
from datar import *
from poison_unlearn import *

def run_mnist_example():
    """
    Example of running MNIST experiments.
    """
    print("Loading MNIST data...")
    xs0, ys0, xs_val, ys_val = mnist_data(500)
    
    print("Running label flipping experiment with QNN...")
    # Run a small experiment for demonstration
    results = qnn_y_flip(xs0, ys0, xs_val, ys_val, ave=1, dep=4, epo=10, bat=256, lr=0.005)
    print(f"Results: {results}")

def run_xxz_example():
    """
    Example of running XXZ model experiments.
    Note: This requires the XXZ dataset files to be present.
    """
    print("Loading XXZ data...")
    try:
        xs0, ys0, xs_val, ys_val = xxz_data()
        
        print("Running label flipping experiment with QNN...")
        # Run a small experiment for demonstration
        results = qnn_y_flip(xs0, ys0, xs_val, ys_val, ave=1, dep=4, epo=10, bat=32, lr=0.03)
        print(f"Results: {results}")
    except FileNotFoundError:
        print("XXZ dataset files not found. Skipping this example.")

if __name__ == "__main__":
    print("Machine Unlearning Examples")
    print("=" * 40)
    
    print("\n1. MNIST Example:")
    run_mnist_example()
    
    print("\n2. XXZ Model Example:")
    run_xxz_example()
    
    print("\nFor full experiments, run:")
    print("python run_unlearn_mnist.py")
    print("python run_unlearn_xxz.py")