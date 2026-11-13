#include "../include/linear.h"

#include "../include/operations.h"

#include <stdexcept>


Tensor Linear::forward(const Tensor& input) const{
    const std::vector<std::size_t>& input_shape = input.get_shape();
    const std::vector<std::size_t>& weight_shape = weights.get_shape();
    if(input_shape.size() < 1){
        throw std::invalid_argument("Input Tensor is not rank 1 or greater");
       }
    if(input_shape.back() != weight_shape[0]){
        throw std::invalid_argument("Incompatible input and weight Tensor shapes");
    }
    Tensor product = matmul(input, weights);
    return add_bias(product, bias);
}