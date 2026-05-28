cmake_minimum_required(VERSION 3.22)

if (POLICY CMP0135)
    # Use download time as timestamp when extracting files from archives
    cmake_policy(SET CMP0135 NEW)
endif()


if(NOT "${LAST_CMAKE_VERSION}" VERSION_EQUAL ${CMAKE_VERSION})
    set(LAST_CMAKE_VERSION ${CMAKE_VERSION} CACHE INTERNAL "Last version of cmake used to configure")
    if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
        message(STATUS "Running CMake version ${CMAKE_VERSION}")
    endif()
endif()


file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION METATENSOR_JAX_VERSION)
string(STRIP ${METATENSOR_JAX_VERSION} METATENSOR_JAX_VERSION)

include(cmake/dev-versions.cmake)
create_development_version("${METATENSOR_JAX_VERSION}" METATENSOR_JAX_FULL_VERSION "metatensor-jax-v")
message(STATUS "Building metatensor-jax v${METATENSOR_JAX_FULL_VERSION}")

# strip any -dev/-rc suffix on the version since project(VERSION) does not support it
string(REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3" METATENSOR_JAX_VERSION ${METATENSOR_JAX_FULL_VERSION})
project(metatensor_jax
    VERSION ${METATENSOR_JAX_VERSION}
    LANGUAGES CXX
)
set(PROJECT_VERSION ${METATENSOR_JAX_FULL_VERSION})

option(METATENSOR_JAX_TESTS "Build metatensor-jax C++ tests" OFF)
include(GNUInstallDirs)

# Set a default build type if none was specified
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(
            CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: none(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) debug release relwithdebinfo minsizerel."
            FORCE
        )
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug relwithdebinfo minsizerel none)
    endif()
endif()

function(check_compatible_versions _actual_ _requested_)
    if(${_actual_} VERSION_LESS ${_requested_})
        message(FATAL_ERROR "Incompatible versions: we need ${_requested_}, but we got ${_actual_}")
    endif()

    if(${_actual_} MATCHES "^([0-9]+)\\.([0-9]+)")
        set(_actual_major_ "${CMAKE_MATCH_1}")
        set(_actual_minor_ "${CMAKE_MATCH_2}")
    else()
        message(FATAL_ERROR "Failed to parse actual version: ${_actual_}")
    endif()

    if(${_requested_} MATCHES "^([0-9]+)\\.([0-9]+)")
        set(_requested_major_ "${CMAKE_MATCH_1}")
        set(_requested_minor_ "${CMAKE_MATCH_2}")
    else()
        message(FATAL_ERROR "Failed to parse requested version: ${_requested_}")
    endif()

    if (${_requested_major_} EQUAL 0 AND ${_actual_minor_} EQUAL ${_requested_minor_})
        # major version is 0 and same minor version, everything is fine
    elseif (${_actual_major_} EQUAL ${_requested_major_})
        # same major version, everything is fine
    else()
        # not compatible
        message(FATAL_ERROR "Incompatible versions: we need ${_requested_}, but we got ${_actual_}")
    endif()
endfunction()


set(REQUIRED_METATENSOR_VERSION "0.1.20")
if (NOT "$ENV{METATENSOR_NO_LOCAL_DEPS}" STREQUAL "1")
    create_development_version("${REQUIRED_METATENSOR_VERSION}" METATENSOR_CORE_FULL_VERSION "metatensor-core-v")
else()
    set(METATENSOR_CORE_FULL_VERSION ${REQUIRED_METATENSOR_VERSION})
endif()
string(REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3" REQUIRED_METATENSOR_VERSION ${METATENSOR_CORE_FULL_VERSION})

# Either metatensor is built as part of the same CMake project, or we try to
# find the corresponding CMake package
if (TARGET metatensor::shared)
    get_target_property(METATENSOR_BUILD_VERSION metatensor::shared BUILD_VERSION)
    check_compatible_versions(${METATENSOR_BUILD_VERSION} ${REQUIRED_METATENSOR_VERSION})
else()
    find_package(metatensor ${REQUIRED_METATENSOR_VERSION} CONFIG REQUIRED)
endif()

# XLA FFI headers ship with jaxlib; XLA_FFI_INCLUDE_DIR is set by the
# /etc/profile.d/metatensor-jax.sh fixture to point at
# <jaxlib>/include where xla/ffi/api/{c_api.h,api.h,ffi.h} sit.
if (NOT DEFINED XLA_FFI_INCLUDE_DIR)
    if (DEFINED ENV{XLA_FFI_INCLUDE_DIR})
        set(XLA_FFI_INCLUDE_DIR "$ENV{XLA_FFI_INCLUDE_DIR}")
    endif()
endif()

if (NOT XLA_FFI_INCLUDE_DIR OR NOT EXISTS "${XLA_FFI_INCLUDE_DIR}/xla/ffi/api/ffi.h")
    message(FATAL_ERROR
        "could not find xla/ffi/api/ffi.h. "
        "Set XLA_FFI_INCLUDE_DIR to <jaxlib>/include (the directory containing xla/ffi/api/)"
    )
endif()
message(STATUS "Using XLA FFI headers from ${XLA_FFI_INCLUDE_DIR}")

set(METATENSOR_JAX_HEADERS
    "include/metatensor/jax/ffi_handlers.hpp"
)

set(METATENSOR_JAX_SOURCES
    "src/register.cpp"
)

add_library(metatensor_jax SHARED
    ${METATENSOR_JAX_HEADERS}
    ${METATENSOR_JAX_SOURCES}
)

set_target_properties(metatensor_jax PROPERTIES
    BUILD_VERSION ${METATENSOR_JAX_FULL_VERSION}
)

target_link_libraries(metatensor_jax PUBLIC metatensor::shared ${CMAKE_DL_LIBS})
target_compile_features(metatensor_jax PUBLIC cxx_std_17)
target_include_directories(metatensor_jax PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${XLA_FFI_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Keep handler-symbol visibility default: XLA_FFI_DEFINE_HANDLER_SYMBOL
# emits the metatensor_<op> extern "C" entry points, and Python loads them
# via ctypes by name. Hiding them would defeat the FFI registration.

include(GenerateExportHeader)
generate_export_header(metatensor_jax
    BASE_NAME METATENSOR_JAX
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/metatensor/jax/exports.h
)
target_compile_definitions(metatensor_jax PRIVATE metatensor_jax_EXPORTS)


set(_path_ "${CMAKE_CURRENT_BINARY_DIR}/generated-version.h")
file(WRITE ${_path_} "#pragma once\n\n")
file(APPEND ${_path_} "/** Full version of metatensor-jax as a string */\n")
file(APPEND ${_path_} "#define METATENSOR_JAX_VERSION \"${METATENSOR_JAX_FULL_VERSION}\"\n\n")
file(APPEND ${_path_} "/** Major version number of metatensor-jax as an integer */\n")
file(APPEND ${_path_} "#define METATENSOR_JAX_VERSION_MAJOR ${PROJECT_VERSION_MAJOR}\n\n")
file(APPEND ${_path_} "/** Minor version number of metatensor-jax as an integer */\n")
file(APPEND ${_path_} "#define METATENSOR_JAX_VERSION_MINOR ${PROJECT_VERSION_MINOR}\n\n")
file(APPEND ${_path_} "/** Patch version number of metatensor-jax as an integer */\n")
file(APPEND ${_path_} "#define METATENSOR_JAX_VERSION_PATCH ${PROJECT_VERSION_PATCH}\n")

set(_destination_ "${CMAKE_CURRENT_BINARY_DIR}/include/metatensor/jax/version.h")
file(COPY_FILE ${_path_} ${_destination_} ONLY_IF_DIFFERENT)

if (METATENSOR_JAX_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    metatensor_jax-config-version.cmake
    VERSION ${METATENSOR_JAX_FULL_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(TARGETS metatensor_jax
    EXPORT metatensor_jax-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(EXPORT metatensor_jax-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metatensor_jax
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/metatensor_jax-config.in.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/metatensor_jax-config.cmake
    @ONLY
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/metatensor_jax-config-version.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/metatensor_jax-config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metatensor_jax
)

install(DIRECTORY "include/metatensor" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/metatensor DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
