cmake_minimum_required(VERSION 3.18)
project(FlashAttention CUDA CXX)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)


set(CMAKE_CUDA_ARCHITECTURES 89)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


set(CMAKE_CUDA_FLAGS         "${CMAKE_CUDA_FLAGS} -O3 -lineinfo -Xptxas -v")
set(CMAKE_CUDA_FLAGS_DEBUG   "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo -Xptxas -v")


find_package(CUDAToolkit REQUIRED)


add_executable(flash_attention
    src/main.cu
    src/naive_attentions.cu
    src/flash_attention.cu
)


target_include_directories(flash_attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

target_link_libraries(flash_attention
    m                        # math library (sqrt, exp, etc.)
    CUDA::cudart             # CUDA runtime
)

message(STATUS "")
message(STATUS "=== Flash Attention Build Configuration ===")
message(STATUS "  CUDA Architecture : sm_${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA flags        : ${CMAKE_CUDA_FLAGS}")
message(STATUS "  Source dir        : ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  Binary output     : ${CMAKE_BINARY_DIR}/flash_attention")
message(STATUS "===========================================")
message(STATUS "")
