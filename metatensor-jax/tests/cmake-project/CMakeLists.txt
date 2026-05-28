cmake_minimum_required(VERSION 3.22)

message(STATUS "Running with CMake version ${CMAKE_VERSION}")

project(metatensor-jax-test-cmake-project CXX)

option(USE_CMAKE_SUBDIRECTORY OFF)

if (USE_CMAKE_SUBDIRECTORY)
    message(STATUS "Using metatensor-jax with add_subdirectory")
    set(BUILD_METATENSOR_JAX ON)
    add_subdirectory(../../../ metatensor)

    set(CMAKE_BUILD_RPATH "$<TARGET_FILE_DIR:metatensor::shared>;$<TARGET_FILE_DIR:metatensor_jax>")
else()
    message(STATUS "Using metatensor-jax with find_package")
    include(../../cmake/dev-versions.cmake)
    set(REQUIRED_METATENSOR_JAX_VERSION "0.1.0")
    create_development_version("${REQUIRED_METATENSOR_JAX_VERSION}" METATENSOR_JAX_FULL_VERSION "metatensor-jax-v")
    string(REGEX REPLACE "([0-9]*)\\.([0-9]*).*" "\\1.\\2" REQUIRED_METATENSOR_JAX_VERSION ${METATENSOR_JAX_FULL_VERSION})

    find_package(metatensor_jax ${REQUIRED_METATENSOR_JAX_VERSION} CONFIG REQUIRED)

    get_target_property(mts_jax_build_version metatensor_jax BUILD_VERSION)
    if (NOT ${mts_jax_build_version} STREQUAL ${METATENSOR_JAX_FULL_VERSION})
        message(FATAL_ERROR "Invalid BUILD_VERSION for metatensor_jax, expected ${METATENSOR_JAX_FULL_VERSION} but got ${mts_jax_build_version}")
    endif()
endif()

add_executable(jax-main src/main.cpp)
target_link_libraries(jax-main metatensor_jax)

enable_testing()
add_test(NAME jax-main COMMAND jax-main)
