include(CMakeFindDependencyMacro)
include(FindPackageHandleStandardArgs)

# use the same version for metatensor-core as the main CMakeLists.txt
set(REQUIRED_METATENSOR_VERSION @REQUIRED_METATENSOR_VERSION@)
find_package(metatensor ${REQUIRED_METATENSOR_VERSION} CONFIG REQUIRED)

include(${CMAKE_CURRENT_LIST_DIR}/metatensor_jax-targets.cmake)

set_target_properties(metatensor_jax PROPERTIES
    BUILD_VERSION "@METATENSOR_JAX_FULL_VERSION@"
)

get_target_property(metatensor_jax_configs metatensor_jax IMPORTED_CONFIGURATIONS)
foreach(config ${metatensor_jax_configs})
    get_target_property(metatensor_jax_library metatensor_jax IMPORTED_LOCATION_${config})
endforeach()

find_package_handle_standard_args(metatensor_jax DEFAULT_MSG metatensor_jax_library)
