#pragma once

#include "tensor.h"

Tensor add(const std::vector<Tensor>& tensors);
Tensor multiply(const std::vector<Tensor>& tensors);
Tensor matmul(const Tensor& A, const Tensor& B);
Tensor relu(const Tensor& input);