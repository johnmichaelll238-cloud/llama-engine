#include <cassert>
#include "../include/operations.h"

int main() {
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

    // Tests for add()

    Tensor y({2, 3, 4});
    y.fill(7.0f);

    Tensor z = add({x, y});

    for (float value : z.get_data()) {
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
    catch (const std::runtime_error&) {
        // expected
    }

    // Tests for multiply()

    Tensor product = multiply({x, y});

    for (float value : product.get_data()) {
        assert(value == 49.0f);
    }

    Tensor product2 = multiply({x, y, z});

    for (float value : product2.get_data()) {
        assert(value == 686.0f);
    }

    // Tests for matmul()

    Tensor A({2, 3});
    Tensor B({3, 2});

    std::vector<float>& A_data = A.get_data();
    std::vector<float>& B_data = B.get_data();

    A_data.assign({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });

    B_data.assign({
         7.0f,  8.0f,
         9.0f, 10.0f,
        11.0f, 12.0f
    });

    Tensor C = matmul(A, B);

    const std::vector<std::size_t> expected_C_shape{2, 2};
    assert((C.get_shape() == expected_C_shape));
    const std::vector<float>& C_data = C.get_data();

    assert(C_data[0] == 58.0f);
    assert(C_data[1] == 64.0f);
    assert(C_data[2] == 139.0f);
    assert(C_data[3] == 154.0f);

   
    // Tests for batched matmul()
    
    Tensor A_batch({2, 3, 4});
    Tensor B_batch({2, 4, 5});
    
    std::vector<float>& A_batch_data = A_batch.get_data();
    std::vector<float>& B_batch_data = B_batch.get_data();
    
    // Batch 0
    A_batch_data = {
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    
        // Batch 1
       13.0f, 14.0f, 15.0f, 16.0f,
       17.0f, 18.0f, 19.0f, 20.0f,
       21.0f, 22.0f, 23.0f, 24.0f
    };
    
    B_batch_data = {
        // Batch 0
         1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
         6.0f,  7.0f,  8.0f,  9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
    
        // Batch 1
        21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
        26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
        31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
        36.0f, 37.0f, 38.0f, 39.0f, 40.0f
    };
    
    Tensor C_batch = matmul(A_batch, B_batch);
    
    const std::vector<std::size_t> expected_C_batch_shape{2, 3, 5};
    assert((C_batch.get_shape() == expected_C_batch_shape));
    
    const std::vector<float>& C_batch_data = C_batch.get_data();
    
    // Batch 0
    assert(C_batch_data[0]  == 110.0f);
    assert(C_batch_data[1]  == 120.0f);
    assert(C_batch_data[2]  == 130.0f);
    assert(C_batch_data[3]  == 140.0f);
    assert(C_batch_data[4]  == 150.0f);
    
    assert(C_batch_data[5]  == 246.0f);
    assert(C_batch_data[6]  == 272.0f);
    assert(C_batch_data[7]  == 298.0f);
    assert(C_batch_data[8]  == 324.0f);
    assert(C_batch_data[9]  == 350.0f);
    
    assert(C_batch_data[10] == 382.0f);
    assert(C_batch_data[11] == 424.0f);
    assert(C_batch_data[12] == 466.0f);
    assert(C_batch_data[13] == 508.0f);
    assert(C_batch_data[14] == 550.0f);
    
    // Batch 1
    assert(C_batch_data[15] == 1678.0f);
    assert(C_batch_data[16] == 1736.0f);
    assert(C_batch_data[17] == 1794.0f);
    assert(C_batch_data[18] == 1852.0f);
    assert(C_batch_data[19] == 1910.0f);
    
    assert(C_batch_data[20] == 2134.0f);
    assert(C_batch_data[21] == 2208.0f);
    assert(C_batch_data[22] == 2282.0f);
    assert(C_batch_data[23] == 2356.0f);
    assert(C_batch_data[24] == 2430.0f);
    
    assert(C_batch_data[25] == 2590.0f);
    assert(C_batch_data[26] == 2680.0f);
    assert(C_batch_data[27] == 2770.0f);
    assert(C_batch_data[28] == 2860.0f);
    assert(C_batch_data[29] == 2950.0f);
    
        

       // Tests batched A with rank-2 B

Tensor A_shared({2, 2, 3});
Tensor B_shared({3, 2});

std::vector<float>& A_shared_data = A_shared.get_data();
std::vector<float>& B_shared_data = B_shared.get_data();

A_shared_data = {
    // Batch 0
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f,

    // Batch 1
    7.0f,  8.0f,  9.0f,
    10.0f, 11.0f, 12.0f
};

B_shared_data = {
    1.0f, 2.0f,
    3.0f, 4.0f,
    5.0f, 6.0f
};

Tensor C_shared = matmul(A_shared, B_shared);

std::vector<std::size_t> expected_shared_shape = {2, 2, 2};
assert((C_shared.get_shape() == expected_shared_shape));

const std::vector<float>& C_shared_data = C_shared.get_data();

// Batch 0
assert(C_shared_data[0] == 22.0f);
assert(C_shared_data[1] == 28.0f);
assert(C_shared_data[2] == 49.0f);
assert(C_shared_data[3] == 64.0f);

// Batch 1
assert(C_shared_data[4] == 76.0f);
assert(C_shared_data[5] == 100.0f);
assert(C_shared_data[6] == 103.0f);
assert(C_shared_data[7] == 136.0f);



// Test 1: Mixed positive, negative, and zero values
Tensor input({5});
input.get_data() = {-2.0f, 0.0f, 3.0f, -4.0f, 5.0f};

Tensor output = relu(input);

std::vector<float> expected = {0.0f, 0.0f, 3.0f, 0.0f, 5.0f};

assert(output.get_data() == expected);

// Test 2: Input must remain unchanged
std::vector<float> original = {-2.0f, 0.0f, 3.0f, -4.0f, 5.0f};

assert(input.get_data() == original);

// Test 3: Output must preserve the input shape
assert(output.get_shape() == input.get_shape());

// Test 4: ReLU must work on rank-three tensors
Tensor input3d({2, 2, 2});
input3d.get_data() = {
    -1.0f, 2.0f,
     3.0f, -4.0f,
     5.0f, -6.0f,
     0.0f, 7.0f
};

Tensor output3d = relu(input3d);

std::vector<float> expected3d = {
    0.0f, 2.0f,
    3.0f, 0.0f,
    5.0f, 0.0f,
    0.0f, 7.0f
};

assert(output3d.get_data() == expected3d);
assert(output3d.get_shape() == input3d.get_shape());

// Test 5: Original rank-three tensor remains unchanged
assert(input3d.get_data() == std::vector<float>({
    -1.0f, 2.0f,
     3.0f, -4.0f,
     5.0f, -6.0f,
     0.0f, 7.0f
}));


    
return 0;
}