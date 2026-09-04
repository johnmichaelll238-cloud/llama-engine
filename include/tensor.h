#include <algorithm>
#include <vector>
#include <iostream>

class Tensor {
    private:
    std::vector<float> data;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;
    
    public:
    Tensor(const std::vector<std::size_t>& shape){
        this->shape = shape;
        std::size_t product = 1;
        for (std::size_t i = 0; i < shape.size(); i++) {
            product *= shape[i];
    }
    data.resize(product);
    
    size_t N = shape.size() - 1;
    std::vector<std::size_t> stride(shape.size());
    std::size_t p_stride = 1;

    for(int i = N; i > -1; i--){
    stride[i] = p_stride;
    p_stride *= shape[i];
    }
     
    this->strides = stride;
}
    const std::vector<std::size_t>& get_shape() const;
    const std::vector<float>& get_data() const;
    std::vector<float>& get_data();
    std::vector<std::size_t> get_strides() const;
    float& at(const std::vector<std::size_t>& coordinates);
    void fill(float value);
    
};