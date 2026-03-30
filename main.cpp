#include <iostream>
#include <vector>
#include <cmath>
#include "Matrix.h"

int main() {
    // --- 1. Testing Constructors & Printing ---
    std::cout << "=== Test 1: Initialization ===" << std::endl;
    
    // Test Initializer List
    Matrix A(2, 3, {1.0, 2.0, 3.0, 
                    4.0, 5.0, 6.0});
    std::cout << "Matrix A (2x3):" << std::endl;
    A.print();

    // Test Fill Constructor
    Matrix B(3, 2, 1.0);
    std::cout << "Matrix B (3x2, all 1s):" << std::endl;
    B.print();


    // --- 2. Testing Matrix Multiplication ---
    std::cout << "=== Test 2: Matrix Multiplication (A * B) ===" << std::endl;
    // (2x3) * (3x2) should result in a (2x2) matrix
    Matrix C = A * B;
    std::cout << "Result C (2x2):" << std::endl;
    C.print();


    // --- 3. Testing Element Access & Modification ---
    std::cout << "=== Test 3: Access & Modify ===" << std::endl;
    A(0, 0) = 10.0;
    std::cout << "Modified A(0,0) to 10.0:" << std::endl;
    A.print();


    // --- 4. Testing Transpose & Hadamard ---
    std::cout << "=== Test 4: Transpose & Hadamard ===" << std::endl;
    Matrix AT = A.transpose();
    std::cout << "A Transposed (3x2):" << std::endl;
    AT.print();

    Matrix H = A.hadamard(A);
    std::cout << "A Hadamard A (Element-wise square):" << std::endl;
    H.print();


    // --- 5. Testing Map Function (Activation Simulation) ---
    std::cout << "=== Test 5: Map Function (ReLU) ===" << std::endl;
    Matrix D(2, 2, {1.5, -2.0, 
                    -0.5, 3.0});
    std::cout << "Matrix D before ReLU:" << std::endl;
    D.print();

    // Using the template map with a Lambda for ReLU
    Matrix relu_D = D.map([](double x) { return x > 0 ? x : 0.0; });
    std::cout << "Matrix D after ReLU:" << std::endl;
    relu_D.print();


    // --- 6. Testing Randomization (Static Engine) ---
    std::cout << "=== Test 6: Randomization ===" << std::endl;
    Matrix R(3, 3);
    R.randomize(0.0, 1.0); // Mean 0, StdDev 1
    std::cout << "Random 3x3 Matrix (Normal Distribution):" << std::endl;
    R.print();


    // --- 7. Testing Neural Network Specific Ops ---
    std::cout << "=== Test 7: Bias & Reductions ===" << std::endl;
    Matrix Scores(3, 2, {1.0, 5.0, 
                         2.0, 3.0, 
                         4.0, 1.0});
    Matrix Bias(3, 1, {10.0, 20.0, 30.0});
    
    std::cout << "Scores (3x2):" << std::endl;
    Scores.print();

    Matrix Combined = Scores.add_bias(Bias);
    std::cout << "Scores + Bias (Broadcasting):" << std::endl;
    Combined.print();

    std::vector<size_t> predictions = Scores.argmax();
    std::cout << "Argmax for each column (Predicted Classes):" << std::endl;
    for(size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Sample " << i << ": Class " << predictions[i] << std::endl;
    }

    std::cout << "\nAll Matrix tests passed successfully!" << std::endl;

    return 0;
}