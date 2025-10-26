# GPNN - Gaussian Process Neural Networks

A Python package for Gaussian Process (GP) regression with PyTorch, featuring various kernel implementations, uncertainty quantification, and automatic hyperparameter optimization through gradient descent.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Multiple Kernels**: RBF (Squared Exponential), Matérn, and Polynomial kernels
- **Automatic Differentiation**: PyTorch-based implementation with learnable hyperparameters
- **GPU Support**: Full PyTorch GPU acceleration for large-scale problems
- **Uncertainty Quantification**: Built-in prediction uncertainty estimates with confidence intervals
- **Robust Numerics**: Automatic jitter adjustment for numerical stability
- **Visualization**: Publication-quality plotting of predictions with uncertainty
- **Modular Design**: Clean, extensible architecture with custom exceptions and configuration

## Installation

### From PyPI (when published)

```bash
pip install gpnn
```

### From Source

```bash
git clone https://github.com/abhimamg/gpnn.git
cd gpnn
pip install .
```

### Development Installation

For development with all dependencies:

```bash
git clone https://github.com/abhimamg/gpnn.git
cd gpnn
uv sync
```

## Requirements

- Python 3.12+
- PyTorch 2.0+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0

## Quick Start

```python
import torch
import numpy as np
import random
from gpnn import GaussianProcess, RBFKernel, plot_gp_predictions

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Generate synthetic data
N = 100
X_full = torch.randn(N, 1)
y_full = torch.sin(X_full * 2.0 * np.pi / 4.0) + torch.randn(N, 1) * 0.1

# Split data
train_size = 80
X_train = X_full[:train_size]
y_train = y_full[:train_size]
X_val = X_full[train_size:]
y_val = y_full[train_size:]

# Initialize GP model with RBF kernel
kernel = RBFKernel(length_scale=1.0, amplitude=1.0)
gp_model = GaussianProcess(kernel=kernel, noise_scale=0.1)

# Train the model (optimize hyperparameters)
optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.05)
epochs = 30

for epoch in range(epochs):
    nll = gp_model.train_step(X_train, y_train, optimizer)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, NLL = {nll:.4f}")

# Make predictions on a grid
grid = torch.linspace(-3, 3, 100).unsqueeze(1)
mu, var = gp_model.predict(grid)

# Visualize results
plot_gp_predictions(X_train, y_train, grid, mu, var,
                   title="GP Regression with RBF Kernel")

# Inspect learned hyperparameters
print(f"\nLearned hyperparameters:")
print(gp_model.get_hyperparameters())
```

## API Reference

### Kernels

All kernels inherit from the base `Kernel` class and implement the covariance function.

#### RBFKernel (Radial Basis Function / Squared Exponential)

Produces smooth, infinitely differentiable functions. Best for modeling smooth phenomena.

```python
from gpnn import RBFKernel

kernel = RBFKernel(
    length_scale=1.0,  # Controls how quickly correlation decays with distance
    amplitude=1.0       # Controls vertical scale of variation
)
```

**Mathematical form**: `k(x, x') = σ² * exp(-||x - x'||² / (2ℓ²))`

**Use cases**: Smooth functions, interpolation, general-purpose regression

#### MaternKernel

More flexible than RBF, allows control over smoothness. This implementation uses ν=3/2 (once differentiable).

```python
from gpnn import MaternKernel

kernel = MaternKernel(
    length_scale=1.0,   # Controls decay rate
    amplitude=1.0       # Controls vertical scale
)
```

**Mathematical form**: `k(x, x') = σ² * (1 + √3·r/ℓ) * exp(-√3·r/ℓ)` where `r = ||x - x'||`

**Use cases**: Rougher functions, time series, signals with discontinuities

#### PolynomialKernel

Non-stationary kernel for polynomial trends.

```python
from gpnn import PolynomialKernel

kernel = PolynomialKernel(
    degree=2,   # Polynomial degree (1=linear, 2=quadratic, etc.)
    alpha=1.0   # Offset term
)
```

**Mathematical form**: `k(x, x') = (α + x·x')^d`

**Use cases**: Polynomial trends, classification, non-stationary patterns

### GaussianProcess Model

The main class for GP regression.

```python
from gpnn import GaussianProcess

gp = GaussianProcess(
    kernel=None,         # Kernel object (defaults to RBFKernel)
    noise_scale=0.1,     # Observation noise (σ)
    jitter=1e-6          # Numerical stability parameter
)
```

#### Methods

**`fit(X, y)`**
Fit the GP to training data and compute marginal log-likelihood.

```python
X_train = torch.randn(100, 3)  # 100 samples, 3 features
y_train = torch.randn(100, 1)  # 100 targets
log_likelihood = gp.fit(X_train, y_train)
```

**`predict(X_test, full_cov=False, return_std=False)`**
Make predictions at test points with uncertainty.

```python
X_test = torch.randn(20, 3)

# Get mean and variance
mean, var = gp.predict(X_test)

# Get mean and standard deviation
mean, std = gp.predict(X_test, return_std=True)

# Get full covariance matrix
mean, cov = gp.predict(X_test, full_cov=True)
```

**`train_step(X, y, optimizer)`**
Perform one gradient descent step to optimize hyperparameters.

```python
optimizer = torch.optim.Adam(gp.parameters(), lr=0.01)
for epoch in range(100):
    loss = gp.train_step(X_train, y_train, optimizer)
```

**`get_hyperparameters()`**
Get current hyperparameter values as a dictionary.

```python
params = gp.get_hyperparameters()
print(params)
# {'noise_scale': 0.1, 'kernel_type': 'RBFKernel', 'length_scale': 1.5, ...}
```

### Visualization Functions

#### `plot_gp_predictions`

Plot predictions with uncertainty for 1D inputs.

```python
from gpnn import plot_gp_predictions

plot_gp_predictions(
    X_train, y_train,    # Training data
    X_test, mean, var,   # Predictions
    title="GP Regression",
    confidence_level=0.95,  # 95% confidence intervals
    figsize=(10, 6),
    save_path='results.png'  # Optional: save to file
)
```

#### `plot_training_metrics`

Plot training loss over epochs.

```python
from gpnn import plot_training_metrics

losses = []
for epoch in range(100):
    loss = gp.train_step(X_train, y_train, optimizer)
    losses.append(loss)

plot_training_metrics(losses, title="Training Loss")
```

## Advanced Usage

### GPU Acceleration

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model and data to GPU
gp = GaussianProcess(kernel=RBFKernel()).to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)

# Training and prediction work the same way
gp.fit(X_train, y_train)
mean, var = gp.predict(X_test.to(device))
```

### Custom Learning Rates and Optimizers

```python
# Different learning rates for kernel and noise
optimizer = torch.optim.Adam([
    {'params': gp.kernel.parameters(), 'lr': 0.1},
    {'params': [gp.log_noise_scale], 'lr': 0.01}
])
```

### Multiple Outputs (Work in Progress)

Currently, GPNN supports single-output regression. For multiple outputs, train separate GPs:

```python
gps = [GaussianProcess(kernel=RBFKernel()) for _ in range(n_outputs)]
for i, gp in enumerate(gps):
    gp.fit(X_train, y_train[:, i:i+1])
```

### Hyperparameter Initialization

Good initial hyperparameters can significantly improve training:

```python
# For data with known scale
data_std = y_train.std()
data_lengthscale = X_train.std()

kernel = RBFKernel(
    length_scale=data_lengthscale.item(),
    amplitude=data_std.item()
)
gp = GaussianProcess(kernel=kernel, noise_scale=0.1 * data_std.item())
```

## Troubleshooting

### Numerical Instability

**Problem**: `NumericalInstabilityError: Cholesky decomposition failed`

**Solutions**:
1. Increase `noise_scale` parameter: `GaussianProcess(noise_scale=0.1)`
2. Scale your input features to similar ranges
3. Remove duplicate or near-duplicate data points
4. Increase `jitter` parameter: `GaussianProcess(jitter=1e-5)`

### Poor Predictions

**Problem**: Model predictions don't match data well

**Solutions**:
1. Train for more epochs
2. Try different learning rates (0.01 - 0.1)
3. Try different kernels (Matérn for rougher functions)
4. Check for appropriate lengthscale initialization
5. Ensure data is properly normalized

### Slow Training

**Problem**: Training is taking too long

**Solutions**:
1. Use GPU acceleration (see above)
2. Reduce training data size (GPs scale O(N³))
3. Use sparse/inducing point methods (future feature)
4. Reduce number of training iterations

## Examples

See the `examples/` directory for more detailed examples:

- `basic_example.py`: Complete walkthrough of GP regression
- (More examples coming soon)

## Mathematical Background

Gaussian Processes define a distribution over functions. Given training data (X, y), the model learns:

1. **Prior**: Assumes functions are drawn from a GP with covariance given by the kernel
2. **Likelihood**: Observations are noisy: `y = f(X) + ε`, where `ε ~ N(0, σ²)`
3. **Posterior**: Conditioning on data gives a GP posterior with closed-form predictive distribution

The predictive distribution at test points X* is:

```
f(X*) | X, y, X* ~ N(μ*, Σ*)

μ* = K(X*, X)[K(X, X) + σ²I]⁻¹y
Σ* = K(X*, X*) - K(X*, X)[K(X, X) + σ²I]⁻¹K(X, X*)
```

where K(·,·) is the kernel function.

## Project Structure

```
gpnn/
├── src/gpnn/
│   ├── __init__.py         # Package initialization and exports
│   ├── kernels.py          # Kernel implementations (RBF, Matérn, Polynomial)
│   ├── models.py           # GaussianProcess model
│   ├── utils.py            # Visualization and utilities
│   ├── config.py           # Configuration and constants
│   └── exceptions.py       # Custom exception classes
├── examples/
│   └── basic_example.py    # Example usage
├── tests/
│   └── test_kernels.py     # Unit tests
├── README.md               # This file
├── LICENSE                 # MIT License
└── pyproject.toml          # Package configuration
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

Please see `CONTRIBUTING.md` for detailed guidelines.

## Citation

If you use GPNN in your research, please cite:

```bibtex
@software{gpnn2025,
  author = {Mamgain, Abhishek},
  title = {GPNN: Gaussian Process Neural Networks with PyTorch},
  year = {2025},
  url = {https://github.com/abhimamg/gpnn}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Inspired by [GPyTorch](https://gpytorch.ai/) and [scikit-learn](https://scikit-learn.org/)
- Mathematical foundations from "Gaussian Processes for Machine Learning" by Rasmussen & Williams

## Contact

- **Author**: Abhishek Mamgain
- **Email**: abhi.mamg@gmail.com
- **GitHub**: [@abhimamg](https://github.com/abhimamg)

## Roadmap

Future planned features:

- [ ] Multi-output GPs
- [ ] Sparse/inducing point methods for large-scale data
- [ ] More kernel implementations (Periodic, Spectral Mixture, etc.)
- [ ] Automatic kernel selection
- [ ] Integration with popular ML frameworks
- [ ] More visualization options (2D contour plots, kernel visualizations)
- [ ] Batch prediction for memory efficiency

## Related Projects

- [GPyTorch](https://gpytorch.ai/): Scalable GPs in PyTorch
- [GPflow](https://www.gpflow.org/): GPs in TensorFlow
- [scikit-learn GPs](https://scikit-learn.org/stable/modules/gaussian_process.html): Simple GP implementation

---

**Note**: This package is under active development. APIs may change in future versions.
