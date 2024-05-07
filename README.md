# SpecNN4PDE
SpecNN4PDE is an under development Python library for solving partial differential equations using spectral methods and neural networks. It consists of the following modules:

- `spectral`: Provides functions for working with spectral methods as described in the book [Spectral Methods: Algorithms, Analysis and Applications](https://link.springer.com/book/10.1007/978-3-540-71041-7) by Shen, Tang, and Wang.
- `nn`: Contains classes for building neural networks, including [random feature method (RFM)](https://global-sci.org/intro/article_detail/jml/21029.html) neural networks, etc.
- `sav`: The [Scalar auxiliary variable (SAV)](https://www.sciencedirect.com/science/article/pii/S002199911730774X) optimizer and its variants.
- `npde`: Functions for solving partial differential equations, e.g., calculating the multivariate derivatives.
- `linalg`: This module primarily focuses on numerical algebra methods.
- `utils`: A collection of utility functions for system and package information retrieval, time measurement, etc.

This project is still in the early stages of development, and the API is subject to change. The library is designed to be used in research and educational settings.

## Dependencies

When you install this library using pip, most dependencies will be automatically handled. However, please note that the `nn`, `sav`, and `npde` module requires PyTorch, which needs to be installed separately.

You can install PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/). Please ensure that you select the correct installation command based on your operating system, package manager, Python version, and the specifications of your CUDA toolkit if you are planning to use PyTorch with GPU support.

If you are not planning to use the `nn`, `sav`, and `npde` module, you do not need to install PyTorch.

## Installation

To install this library, you can use pip:

```bash
pip install specnn4pde