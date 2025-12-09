# Data Descriptions

## 1. Noise Robustness Experiment Data

The following files contain experimental results of model performance under different types of noise (**Label Flipping** / **Feature Randomization**) as the noise ratio α*α* varies.

**File names:**

- `fig2c_mnist_phase.npy`
- `fig2d_xxz_phase.npy`

**Data shape:** `(4, 50)`

**Dimension details:**

- **Axis 0 (Settings):** Experimental configurations (4 in total)
  - `0`: MLP with label flipping
  - `1`: QNN with label flipping
  - `2`: MLP with feature randomization
  - `3`: QNN with feature randomization
- **Axis 1 (Data Points):** Flattened dimension combining noise levels and independent runs (50 points total)
  Structure: **10 noise ratios** × **5 independent runs**
- \*_Noise ratios (α):_ From 0.0 to 0.9 in steps of 0.1 (i.e., [0.0, 0.1, ..., 0.9]).

> **Note:** For analysis, reshape this axis to `(10, 5)`—10 noise levels by 5 runs—to compute statistics (e.g., mean and standard deviation) across runs for each noise level.

## 2. Unlearning Experiment Data

The following files contain unlearning trajectory data for both MLP and QNN models across different datasets.

**File names:**

- `fig3a_mnist_mlp_y.npy` / `fig3a_mnist_qnn_y.npy`
- `fig3b_xxz_mlp_y.npy` / `fig3b_xxz_qnn_y.npy`

**Data shape:** `(4, 5, 50)`

**Dimension details:**

- **Axis 0 (Methods):** Unlearning methods (4 in total)
  - `0`: Retrain
  - `1`: Finetune
  - `2`: Scrub
  - `3`: Grad-Asc (Gradient Ascent)
- **Axis 1 (Runs):** Independent experimental repetitions
  - 5 runs total (used for computing mean and standard deviation)
- **Axis 2 (Steps):** Training steps
  - 50 steps recorded (validation accuracy at steps 0 through 49)
