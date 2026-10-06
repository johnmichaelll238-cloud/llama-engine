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

Tensor matmul(const Tensor& A, const Tensor& B) {

    const std::vector<std::size_t>& A_shape = A.get_shape();
    const std::vector<std::size_t>& B_shape = B.get_shape();

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


        

