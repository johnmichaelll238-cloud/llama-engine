cmake_minimum_required(VERSION 3.16)

project(llama_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tensor_tests
    src/tensor.cpp
    src/operations.cpp
    tests/tensor_tests.cpp
)
target_include_directories(tensor_tests
    PRIVATE
    include
)