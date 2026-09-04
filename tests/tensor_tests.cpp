#include <cassert>
#include "../include/operations.h"


int main(){
Tensor x({2, 3, 4});

std::vector<std::size_t> shape = x.get_shape();
assert(shape.size() == 3);
assert(shape[0] == 2);
assert(shape[1] == 3);
assert(shape[2] == 4);

std::vector<float> data = x.get_data();
assert(data.size() == 24);

std::vector<std::size_t> strides = x.get_strides();

assert(strides.size() == 3);
assert(strides[0] == 12);
assert(strides[1] == 4);
assert(strides[2] == 1);

x.at({1, 2, 3}) = 42.0f;
assert(x.get_data()[23] == 42.0f);

x.fill(7.0f);
std::vector<float> data2 = x.get_data();
for (float value : data2) {
    assert(value == 7.0f);
}
//I'll add the tests for the add() method here:
Tensor y({2, 3, 4});

y.fill(7.0f);

Tensor z = add({x, y});

for (float value : z.get_data()){
    assert(value == 14.0f);
}

try {
    add({x});
    assert(false);
}
catch (const std::runtime_error&) {
    // expected
}

Tensor w({2, 3, 5});

try {
    add({x, w});
    assert(false);
}
catch (const std::runtime_error&){
    
}

}

