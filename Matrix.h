#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cassert>
#include <initializer_list>
#include <functional>

class Matrix{
    
private:
    // Flattened 1D storage for performance
    std::vector<double> data_;
    size_t rows_;
    size_t cols_;
    
    // Internal helper to calculate 1D index from 2D coordinates
    size_t get_idx(size_t row, size_t col) const{
        return row * cols_ + col;
    }

    // Ensures index is within valid range during Debug mode
    void check_bounds(size_t row, size_t col) const{
        assert(row < rows_ && col < cols_ && "Matrix index out of bounds!");
    }

    // Ensures two matrices have identical shapes for element-wise operations
    void check_dimensions(const Matrix& other) const{
        assert(rows_ == other.rows_ && cols_ == other.cols_ && "Matrix dimensions must match!");
    }
public:
    // --- Constructors ---

    // Default constructor: creates an empty 0x0 matrix
    Matrix();

    // Fill constructor: creates a rows x cols matrix filled with 'val'
    Matrix(size_t rows, size_t cols, double val = 0.0);

    // List constructor: allows Matrix m(2, 2, {1, 2, 3, 4})
    Matrix(size_t rows, size_t cols, std::initializer_list<double> list);

    // --- Basic Getters ---
    size_t rows() const {return rows_;}
    size_t cols() const {return cols_;}
    bool empty() const {return data_.empty();}

    // Total elements: return the total num of elements in the matrix
    size_t size() const {return data_.size();}

    // --- Element Access Operators ---

    // Non-const version: allows modification (e.g., mat(i, j) = 5.0)
    double& operator()(size_t row, size_t col) {
        check_bounds(row, col);
        return data_[get_idx(row, col)];
    }

    // Const version: read-only access
    const double& operator()(size_t row, size_t col) const {
        check_bounds(row, col);
        return data_[get_idx(row, col)];
    }

    // --- Utilities ---

    // Print utility: outputs the matrix to the console in a readable 2D grid
    void print() const;

    // Randomize utility: fills the matrix with normally distributed random numbers
    // stddev -> Standart Deviation
    void randomize(double mean = 0.0, double stddev = 1.0);
    
    // --- Math Operations ---

    // Addition: returns a new matrix as the sum of this and another matrix
    Matrix operator+(const Matrix& other) const;

    // Multiplication: performs matrix-matrix multiplication (dot product)
    Matrix operator*(const Matrix& other) const;

    // Transpose: returns a new matrix with rows and columns swapped
    Matrix transpose() const;

    // Hadamard Product: performs element-wise multiplication
    Matrix hadamard(const Matrix& other) const;

    // Subtraction: returns a new matrix as the difference of this and another matrix
    Matrix operator-(const Matrix& other) const;

    // --- Compound Assignment Operations
    
    // in-place addition: adds another matrix directly to this one
    Matrix& operator+=(const Matrix& other);

    // in-place subtraction: subtracts another matrix directly from this one
    Matrix& operator-=(const Matrix& other);

    // in-place scalar multiplication: multiplies every elements by a constant natively
    Matrix& operator*=(double scalar);

    // in-place scalar division: divides every elements by a constant natively
    Matrix& operator/=(double scalar);

    // --- Functional Operations ---

    // Map utility: applies a function to every element and returns a new matrix.
    // Implemented as a template here in the header for zero-overhead inlining.
    template <typename Func>
    Matrix map(Func func) const {
        Matrix res(rows_, cols_);
        size_t len = data_.size();
        for(size_t i = 0; i < len; i++){
            res.data_[i] = func(this->data_[i]);
        }
        return res;
    }

    // --- Scalar Operations ---
    
    // Scalar addition: adds a constant to every element
    Matrix operator+(double scalar) const;
    
    // Scalar subtraction: subtracts a constant from every element
    Matrix operator-(double scalar) const;
    
    // Scalar multiplication: multiplies every element by a constant
    Matrix operator*(double scalar) const;
    
    // Scalar division: divides every element by a constant
    Matrix operator/(double scalar) const;

    // --- Broadcasting & Reductions ---

    // Bias addition: adds a column vector (bias) to every column of this matrix
    Matrix add_bias(const Matrix& bias) const;

    // Sum columns: collapses all columns into a single column vector (rows_ x 1)
    // Very useful for accumulating bias gradients across a batch of samples.
    Matrix sum_along_cols() const;

    // Argmax: returns the row index of the maximum value for each column.
    // Essential for evaluating model accuracy (finding the most likely class).
    std::vector<size_t> argmax() const;
};

// --- Non-member Operations--- 
// Left-scalar multiplication: allows numbers * Matrix
inline Matrix operator*(double scalar, const Matrix& rhs) {return rhs * scalar;}

#endif