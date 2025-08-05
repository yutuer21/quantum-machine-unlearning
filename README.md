# Machine Unlearning Research Code

This repository contains code for machine unlearning experiments, including:

1. Data poisoning and visualization
2. Hessian-based analysis of model landscapes
3. Various unlearning methods comparison (Retrain, Finetune, Scrub, Grad-Asc)
4. Evaluation of unlearning effectiveness on both classical MLP and quantum neural networks

## File Structure

- `datar.py`: Data preprocessing and loading functions
- `models.py`: Model definitions for both MLP and QNN
- `poison_unlearn.py`: Implementation of various unlearning methods
- `run_unlearn_mnist.py`: Main script for MNIST unlearning experiments
- `run_unlearn_xxz.py`: Main script for XXZ model unlearning experiments
- `test_mnist.ipynb`: Jupyter notebook for MNIST experiments visualization
- `test_xxz.ipynb`: Jupyter notebook for XXZ model experiments visualization
- `heissian.ipynb`: Hessian-based analysis of model landscapes
- `run_example.py`: Example script demonstrating how to run the code

## Requirements

- Python 3.x
- TensorFlow
- JAX
- TensorCircuit
- Optax
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn

Install all requirements with:
```bash
pip install -r requirements.txt
```

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

Note: The XXZ experiments require specific dataset files that are not included in this repository.

### Visualization

The results can be visualized using the provided Jupyter notebooks:
- `test_mnist.ipynb`: For MNIST experiment results
- `test_xxz.ipynb`: For XXZ model experiment results
- `heissian.ipynb`: For Hessian-based analysis

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