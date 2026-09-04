#include "../include/operations.h"

Tensor add(const std::vector<Tensor>& tensors){
    if (tensors.size() < 2) {
        throw std::runtime_error("Addition requires two or more tensors");
        }
    size_t length = tensors[0].get_data().size();
    const std::vector<std::size_t>& shape = tensors[0].get_shape();

    for(const Tensor& tensor : tensors){
        if(tensor.get_shape() != shape){
            throw std::runtime_error("Not all tensors are the same shape, please refer to first tensor");
        }
    }   

    Tensor result(shape);
    std::vector<float>& result_data = result.get_data();
    for(const Tensor& tensor : tensors){
        const std::vector<float>& tensor_data = tensor.get_data();
       for(size_t i = 0; i < length; i++){
          result_data[i] += tensor_data[i];
       }
    }
    
    return result;
}


