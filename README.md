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
&nbsp;   // Create a 2x3 matrix and initialize with random weights
&nbsp;   Matrix weights(2, 3);
&nbsp;   weights.randomize(0.0, 1.0); // Mean 0, StdDev 1.0

&nbsp;   // Create a 3x1 input vector (represented as a matrix)
&nbsp;   Matrix input(3, 1, {0.5, -1.2, 0.8});

&nbsp;   // Forward pass: Z = W * X
&nbsp;   Matrix output = weights * input;

&nbsp;   // Apply ReLU activation function using the template map
&nbsp;   Matrix activated = output.map([](double x) {
&nbsp;       return x > 0 ? x : 0.0;
&nbsp;   });

&nbsp;   std::cout << "Layer Output:" << std::endl;
&nbsp;   activated.print();

&nbsp;   return 0;
}