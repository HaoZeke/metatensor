// Minimal smoke test for libmetatensor_jax: only checks that the shared
// library links cleanly against metatensor::shared and the public visibility
// header, mirroring metatensor-torch/tests/cmake-project/src/main.cpp.

#include <metatensor.h>
#include "metatensor/jax/ffi_handlers.hpp"

int main() {
    // touch the exports header so the linker pulls libmetatensor_jax in
    auto* sym = reinterpret_cast<void*>(&metatensor_jax::metatensor_labels_clone);
    return sym == nullptr ? 1 : 0;
}
