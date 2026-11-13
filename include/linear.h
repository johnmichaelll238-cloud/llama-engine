#pragma once

#include "tensor.h"

#include <vector>

#include <stdexcept>

class Linear{
    private:
    Tensor weights;
    Tensor bias;

    public:
    Linear(const Tensor& weights, const Tensor& bias)
     : weights(weights), bias(bias)
    {
       const std::vector<std::size_t>& weight_shape = weights.get_shape();
       const std::vector<std::size_t>& bias_shape = bias.get_shape();
       if(weight_shape.size() != 2){
        throw std::invalid_argument("Weight Tensor is not rank 2");
       }
       
       if(bias_shape.size() != 1){
        throw std::invalid_argument("Bias is not rank 1");
       }
        if(weight_shape.back() != bias_shape[0]){
            throw std::invalid_argument("Incompatible Weights and Bias");
       }
    }

    Tensor forward(const Tensor& input) const;
};