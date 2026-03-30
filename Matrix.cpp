#include "Matrix.h"
#include <iostream>
#include <iomanip>
#include <random>

Matrix::Matrix() :
    rows_(0),
    cols_(0) {
    // data_ is automatically initialized as an empty vector
}

Matrix::Matrix(size_t rows, size_t cols, double val) :
    rows_(rows),
    cols_(cols),
    data_(rows * cols, val) {
}

Matrix::Matrix(size_t rows, size_t cols, std::initializer_list<double> list) :
    rows_(rows),
    cols_(cols) { 
    assert(list.size() == rows_ * cols_ && "Initializer list size does not match matrix dimensions!");
    data_ = list;
}

void Matrix::print() const {
    for(size_t i = 0; i < rows_; i++){
        for(size_t j = 0; j < cols_; j++){
            std::cout << std::setw(8) << std::setprecision(4) << (*this)(i, j) << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

void Matrix::randomize(double mean, double stddev){
    // Setup random number generator (Mersenne Twister)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, stddev);

    // Fill the flattened 1D array in-place
    for (double& val : data_) {
        val = dist(gen);
    }
}

Matrix Matrix::operator*(const Matrix& other) const {
    assert(cols_ == other.rows_ && "Matrix multiplication dimension mismatch: A.cols must equal B.rows!");
    Matrix res(rows_, other.cols_);
    for(size_t i = 0; i < rows_; i++){
        for(size_t k = 0; k < cols_; k++){
            double tmp = (*this)(i, k);
            for(size_t j = 0; j < other.cols_; j++){
                res(i, j) += tmp * other(k, j);
            }
        }
    }
    return res;
}

Matrix Matrix::operator+(const Matrix& other) const {
    check_dimensions(other);
    Matrix res(rows_, cols_);
    size_t len = data_.size();
    for(size_t i = 0; i < len; i++){
        res.data_[i] = this->data_[i] + other.data_[i];
    }
    return res;
}

Matrix Matrix::operator-(const Matrix& other) const {
    check_dimensions(other);
    Matrix res(rows_, cols_);
    size_t len = data_.size();
    for(size_t i = 0; i < len; i++){
        res.data_[i] = this->data_[i] - other.data_[i];
    }
    return res;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    check_dimensions(other);
    Matrix res(rows_, cols_);
    size_t len = data_.size();
    for(size_t i = 0; i < len; i++){
        res.data_[i] = this->data_[i] * other.data_[i];
    }
    return res;
}

Matrix Matrix::transpose() const {
    Matrix res(cols_, rows_);
    for(size_t i = 0; i < cols_; i++){
        for(size_t j = 0; j < rows_; j++){
            res(i, j) = (*this)(j, i);
        }
    }
    return res;
}

Matrix Matrix::operator+(double scalar) const {
    Matrix res(rows_, cols_);
    size_t len = data_.size();
    for(size_t i = 0; i < len; i++){
        res.data_[i] = this->data_[i] + scalar;
    }
    return res;
}

Matrix Matrix::operator-(double scalar) const {
    Matrix res(rows_, cols_);
    size_t len = data_.size();
    for(size_t i = 0; i < len; i++){
        res.data_[i] = this->data_[i] - scalar;
    }
    return res;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix res(rows_, cols_);
    size_t len = data_.size();
    for(size_t i = 0; i < len; i++){
        res.data_[i] = this->data_[i] * scalar;
    }
    return res;
}

Matrix Matrix::operator/(double scalar) const {
    // Relying on IEEE 754 standard for division by zero (results in inf or NaN)
    Matrix res(rows_, cols_);
    size_t len = data_.size();
    for(size_t i = 0; i < len; i++){
        res.data_[i] = this->data_[i] / scalar;
    }
    return res;
}

Matrix Matrix::add_bias(const Matrix& bias) const {
    assert(bias.cols_ == 1 && bias.rows_ == rows_ && "Bias must be a column vector matching the rows of the matrix!");
    Matrix res(rows_, cols_);
    for(size_t i = 0; i < rows_; i++){
        for(size_t j = 0; j < cols_; j++){
            res(i, j) = (*this)(i, j) + bias(i, 0);
        }
    }
    return res;
}

Matrix Matrix::sum_cols() const {
    // Returns a column vector (rows_ x 1)
    Matrix res(rows_, 1);
    for(size_t i = 0; i < rows_; i++){
        double sum = 0.0;
        for(size_t j = 0; j < cols_; j++){
            sum += (*this)(i, j);
        }
        res(i, 0) = sum;
    }
    return res;
}

std::vector<size_t> Matrix::argmax() const {
    assert(rows_ > 0 && cols_ > 0 && "Cannot perform argmax on an empty matrix!");
    std::vector<size_t> res(cols_, 0);
    
    // Iterate through each column (representing each sample in a batch)
    for(size_t j = 0; j < cols_; j++){
        double max_val = (*this)(0, j);
        size_t max_idx = 0;
        
        // Find the maximum value and its row index in the current column
        for(size_t i = 1; i < rows_; i++){
            if((*this)(i, j) > max_val){
                max_val = (*this)(i, j);
                max_idx = i;
            }
        }
        res[j] = max_idx;
    }
    return res;
}