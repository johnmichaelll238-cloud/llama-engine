#include "../include/tensor.h"
#include <stdexcept>

const std::vector<std::size_t>& Tensor::get_shape() const {
    return shape;
}
const std::vector<float>& Tensor::get_data() const{
    return data;
}
std::vector<float>& Tensor::get_data() {
    return data;
}

std::vector<std::size_t> Tensor::get_strides() const{
    return this->strides;
}

float& Tensor::at(const std::vector<std::size_t>& coordinates){
    if (coordinates.size() != shape.size()){
       throw std::runtime_error("Invalid Number of Coordinates");
    }

    for(std::size_t i = 0; i < shape.size(); i++){
        if (coordinates[i] >= shape[i]){
            throw std::runtime_error("Coordinates are out of bound");
        }
    }

    std::size_t index = 0;
    for(std::size_t i = 0; i < shape.size(); i++){
        index += coordinates[i] * strides[i];
    }

    return data[index];
}


void Tensor::fill(float value){
    for(float& val : data){
     val = value;
    }
}

