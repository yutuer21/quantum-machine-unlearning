# Quantum Machine Unlearning Research Code

This repository contains code for machine unlearning experiments, including:

1. Data poisoning and visualization
2. Hessian-based analysis of model landscapes
3. Various unlearning methods comparison (Retrain, Finetune, Scrub, Grad-Asc)
4. Evaluation of unlearning effectiveness on both classical MLP and quantum neural networks

The paper reference is https://arxiv.org/abs/2508.02422 where we for the first time proposed the concept of **Quantum Machine Unlearning**.

## File Structure

- `datar.py`: Data preprocessing and loading functions
- `models.py`: Model for traning and unlearning definitions for both MLP and QNN
- `poison_unlearn.py`: Implementation of various unlearning method wrappers
- `run_unlearn_mnist.py`: Main script for MNIST unlearning experiments
- `run_unlearn_xxz.py`: Main script for XXZ model unlearning experiments
- `run_example.py`: Run the code with only data poisoning part
- `run_test.py`: A simplified test script for quick verification on functionality parallel `run_unlearn_mnist.py`
- `plot_*.ipynb`: Visualization related notebooks

## Requirements

Install all requirements with (python 3.12 suggested):

```bash
pip install -r requirements.txt
```

This work is enabled by the high performance quantum-classical hybrid software infrastructure of [TensorCircuit-NG](https://github.com/tensorcircuit/tensorcircuit-ng).

## Usage

### Quick Start

Run the example script to test the code:

```bash
python run_example.py
```

### Full Experiments

To run the MNIST unlearning experiments:

```bash
python run_unlearn_mnist.py
```

To run the XXZ model unlearning experiments:

```bash
python run_unlearn_xxz.py
```

### Notes

- No specified hardware is required.

- The typical environment setup and software installation time is around 10 minutes.

- Part of the expected output from the demo `run_example.py` (expected runtime on a desktop: ~5min):

```bash
Machine Unlearning Examples
========================================

1. MNIST Example:
Loading MNIST data...
Training: (500, 784)
250 250
Validation: (1000, 784)
Running label flipping experiment with QNN...
F=5.71 C=6.53 S=12.00 P=13.19: 100%|████████████| 64/64 [00:01<00:00, 43.69it/s]

Results: [{'depth': 4, 'epochs': 10, 'learning_rate': 0.005, 'batch_size': 256, 'acc': Array(0.624, dtype=float32), 'val_acc': Array(0.591, dtype=float32)}, {'depth': 4, 'epochs': 10, 'learning_rate': 0.005, 'batch_size': 256, 'acc': Array(0.67, dtype=float32), 'val_acc': Array(0.69100004, dtype=float32)}, {'depth': 4, 'epochs': 10, 'learning_rate': 0.005, 'batch_size': 256, 'acc': Array(0.70400006, dtype=float32), 'val_acc': Array(0.744, dtype=float32)}, {'depth': 4, 'epochs': 10, 'learning_rate': 0.005, 'batch_size': 256, 'acc': Array(0.57000005, dtype=float32), 'val_acc': Array(0.67200005, dtype=float32)}, {'depth': 4, 'epochs': 10, 'learning_rate': 0.005, 'batch_size': 256, 'acc': Array(0.476, dtype=float32), 'val_acc': Array(0.47200003, dtype=float32)}, {'depth': 4, 'epochs': 10, 'learning_rate': 0.005, 'batch_size': 256, 'acc': Array(0.54200006, dtype=float32), 'val_acc': Array(0.513, dtype=float32)}, {'depth': 4, 'epochs': 10, 'learning_rate': 0.005, 'batch_size': 256, 'acc': Array(0.522, dtype=float32), 'val_acc': Array(0.48700002, dtype=float32)}, {'depth': 4, 'epochs': 10, 'learning_rate': 0.005, 'batch_size': 256, 'acc': Array(0.59400004, dtype=float32), 'val_acc': Array(0.34, dtype=float32)}, {'depth': 4, 'epochs': 10, 'learning_rate': 0.005, 'batch_size': 256, 'acc': Array(0.614, dtype=float32), 'val_acc': Array(0.27100003, dtype=float32)}, {'depth': 4, 'epochs': 10, 'learning_rate': 0.005, 'batch_size': 256, 'acc': Array(0.67800003, dtype=float32), 'val_acc': Array(0.261, dtype=float32)}]

2. XXZ Model Example:
Loading XXZ data...
```

## Methods

The repository implements several machine unlearning methods:

1. **Retrain**: Retrain the model from scratch on the retained data
2. **Fine-tune (cf)**: Continue training on the retained data
3. **Scrub**: Use KL divergence regularization to "scrub" the influence of forgotten data
4. **Gradient Ascent (ga)**: Use gradient ascent to actively unlearn the forgotten data

## Models

Two types of models are implemented:

1. **Classical MLP**: Multi-layer perceptron with configurable hidden layers
2. **Quantum Neural Network (QNN)**: Quantum circuit-based model using TensorCircuit

## Data

The code supports experiments on:

1. **MNIST**: Binary classification of digits 1 and 9
2. **XXZ Model**: Quantum many-body system data (requires external dataset files)

## Results

The experiments evaluate:

1. **Validation Accuracy**: Performance on clean validation data
2. **Forgetting Accuracy**: Ability to forget the "unlearned" data
3. **Model Stability**: Robustness of the unlearning process

## License

This project is licensed under the MIT License - see the LICENSE file for details.
