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

Tensor multiply(const std::vector<Tensor>& tensors){
    if (tensors.size() < 2) {
        throw std::runtime_error("Multiplication requires two or more tensors");
        }
    size_t length = tensors[0].get_data().size();
    const std::vector<std::size_t>& shape = tensors[0].get_shape();

    for(const Tensor& tensor : tensors){
        if(tensor.get_shape() != shape){
            throw std::runtime_error("Not all tensors are the same shape, please refer to first tensor");
        }
    }   

    Tensor result(shape);
    result.fill(1.0f);
    std::vector<float>& result_data = result.get_data();
    for(const Tensor& tensor : tensors){
        const std::vector<float>& tensor_data = tensor.get_data();
       for(size_t i = 0; i < length; i++){
          result_data[i] *= tensor_data[i];
       }
    }
    
    return result;
}

void iterate(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    const std::vector<std::size_t>& batch_shape,
    std::vector<std::size_t>& coords,
    std::size_t dimension
){
    if (dimension >= batch_shape.size()) {
        const std::vector<std::size_t>& A_shape = A.get_shape();
        const std::vector<std::size_t>& B_shape = B.get_shape();
        
        const std::size_t M = A_shape[A_shape.size() - 2];
        const std::size_t K = A_shape[A_shape.size() - 1];
        const std::size_t N = B_shape[B_shape.size() - 1];
        
        for (std::size_t row = 0; row < M; ++row) {
            for (std::size_t k = 0; k < K; ++k) {
                for (std::size_t col = 0; col < N; ++col) {
        
                    std::vector<std::size_t> A_coords = coords;
                    A_coords.push_back(row);
                    A_coords.push_back(k);
        
                    std::vector<std::size_t> B_coords;

                    if (B_shape.size() == 2) {
                        B_coords.push_back(k);
                        B_coords.push_back(col);
                    } else {
                        B_coords = coords;
                        B_coords.push_back(k);
                        B_coords.push_back(col);
                    }
        
                    std::vector<std::size_t> C_coords = coords;
                    C_coords.push_back(row);
                    C_coords.push_back(col);
        
                    C.at(C_coords) +=
                        A.at(A_coords) * B.at(B_coords);
                }
            }
        }
        return;
    }

    for (std::size_t i = 0; i < batch_shape[dimension]; i++) {
          coords[dimension] = i;
        iterate(A, B, C, batch_shape, coords, dimension + 1);
    }
}

Tensor vector_matrix(const Tensor& A, const Tensor& B){

    //valiation of operand shapes
    
    const std::vector<std::size_t>& A_shape = A.get_shape();
    const std::vector<std::size_t>& B_shape = B.get_shape();
    if(A_shape.size() != 1){
     throw std::invalid_argument("First operand does not have shape or dimensionality of '1'");
    }
    if(B_shape.size() != 2){
    throw std::invalid_argument("Second operand does not have shape or dimensionality of '2'");
    }
    
    //validation of shape compatiility
    
    if(A_shape[0] != B_shape[0]){
     throw std::invalid_argument("Shapes are incompatible, no 'k' to sum over");
    }
    
    //create empty Tensor that will be given as the return value
    std::vector<std::size_t> result_shape = {B_shape[1]};
    Tensor result(result_shape);
    result.fill(0.0f);
    
    // write loop for cumulating elements in the result Tensor over shared dimension "K"
    std::size_t k = A_shape[0];
    std::size_t N = B_shape[1];
    
    for(std::size_t i = 0; i < k; i++){
       for(std::size_t j = 0; j < N; j++){
          result.at({j}) += A.at({i}) * B.at({i, j});
    }
    }
    
    //return result Tensor
    
    return result;
    }
    

Tensor matmul(const Tensor& A, const Tensor& B) {

    const std::vector<std::size_t>& A_shape = A.get_shape();
    const std::vector<std::size_t>& B_shape = B.get_shape();
    
    if(A_shape.size() == 1 && B_shape.size() == 2){
        return vector_matrix(A, B);
  
    }    

    if (A_shape.size() < 2 || B_shape.size() < 2) {
        throw std::runtime_error(
            "Shapes are too small for matrix multiplication"
        );
    }

    // B may either:
    // 1. Have the same rank as A, with matching batch dimensions
    // 2. Be rank 2, meaning the same matrix is used for every A batch
    if (B_shape.size() != 2 &&
        A_shape.size() != B_shape.size()) {

        throw std::runtime_error(
            "Shapes are not compatible for matrix multiplication"
        );
    }

    std::vector<std::size_t> batch_shape(
        A_shape.begin(),
        A_shape.end() - 2
    );

    std::vector<std::size_t> coords(batch_shape.size(), 0);

    // Only compare batch dimensions when B actually has batch dimensions.
    if (B_shape.size() > 2) {
        for (std::size_t i = 0; i < batch_shape.size(); ++i) {
            if (A_shape[i] != B_shape[i]) {
                throw std::runtime_error(
                    "Batch dimensions do not match"
                );
            }
        }
    }

    // K dimension must always match.
    if (A_shape[A_shape.size() - 1] !=
        B_shape[B_shape.size() - 2]) {

        throw std::runtime_error(
            "Invalid shapes for matrix multiplication, "
            "no 'K' dimension to sum over"
        );
    }

    std::vector<std::size_t> c_shape = batch_shape;

    c_shape.push_back(A_shape[A_shape.size() - 2]);
    c_shape.push_back(B_shape[B_shape.size() - 1]);

    Tensor C(c_shape);
    C.fill(0.0f);

    iterate(A, B, C, batch_shape, coords, 0);

    return C;
}

Tensor relu(const Tensor& input){
    Tensor output = input;
    std::vector<float>& data = output.get_data();
    for(std::size_t i = 0; i < data.size(); i++){
        if(data[i] < 0){
            data[i] = 0;
        }
    }
    return output;
}

Tensor add_bias(const Tensor& input, const Tensor& bias){  
    std::vector<std::size_t> input_shape = input.get_shape();
    const std::vector<float>& input_data = input.get_data();
    std::vector<std::size_t> bias_shape = bias.get_shape();
    const std::vector<float>& bias_data = bias.get_data();
    //parameter validation:
    if(input_shape.size() < 1){
    throw std::runtime_error("Initial vector is not the right rank - [Not rank 1 or greater]");
    }
     if(bias_shape.size() != 1){
    throw std::runtime_error("Bias tensor is not the right rank");
    }
    if(input_shape.back() != bias_shape[0]){
    throw std::runtime_error("Incompatible tensor shapes for bias addition");
    }
    //Create candidate result Tensor:
    
    Tensor result(input_shape);
    std::vector<float>& rdata = result.get_data();
    
    //Commence bias addition:
    std::size_t i = 0;
    while(i < input_data.size()){
      for(std::size_t j = 0; j < bias_data.size(); j++){
       rdata[i] = input_data[i] + bias_data[j];
       i++;
    }
    
    }
    
    return result;
    }       

