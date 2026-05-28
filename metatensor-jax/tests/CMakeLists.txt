enable_testing()

add_executable(jax-main cmake-project/src/main.cpp)
target_link_libraries(jax-main metatensor_jax)
add_test(NAME jax-main COMMAND jax-main)
