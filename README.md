# Matrix-Cpp

A high-performance, lightweight matrix library implemented in C++ from scratch, specifically engineered for building neural networks (MLP) and practicing low-level computational optimization.

## 🚀 Features

* **1D Flattened Storage:** Utilizes a single `std::vector<double>` for row-major storage to maximize CPU cache hits and ensure memory continuity.
* **Optimized Matrix Multiplication:** Implements the **i-k-j loop order** optimization to significantly reduce Cache Misses compared to naive implementations.
* **Zero-Overhead Functional Mapping:** * Features a **Template-based `map` function** that allows passing lambdas or custom activation functions (ReLU, Sigmoid).
  * Leverages compiler inlining to eliminate the overhead of function pointers or `std::function`.
* **MLP-Ready Operations:** Built-in support for essential deep learning math:
  * **Hadamard Product:** Element-wise multiplication.
  * **Broadcasting Bias:** Efficiently adds column vectors (biases) to every column of a matrix.
  * **Reductions:** `sum_cols()` for gradient accumulation and `argmax()` for classification prediction.
* **Statistical Initialization:** Includes a high-quality randomization engine based on the **Mersenne Twister (MT19937)** for normal distribution weight initialization.

## 💻 Usage Example

```cpp
#include <iostream>
#include "Matrix.h"

int main() {
    // Create a 2x3 matrix and initialize with random weights
    Matrix weights(2, 3);
    weights.randomize(0.0, 1.0); // Mean 0, StdDev 1.0

    // Create a 3x1 input vector (represented as a matrix)
    Matrix input(3, 1, {0.5, -1.2, 0.8});

    // Forward pass: Z = W * X
    Matrix output = weights * input;

    // Apply ReLU activation function using the template map
    Matrix activated = output.map([](double x) {
       return x > 0 ? x : 0.0;
    });

    std::cout << "Layer Output:" << std::endl;
    activated.print();

    return 0;
}