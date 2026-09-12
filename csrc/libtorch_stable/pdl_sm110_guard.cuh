// SPDX-License-Identifier: Apache-2.0
// PDL launch attributes can hang at cudaGridDependencySynchronize() on Tegra
// SM110 (unified memory): the kernel spins waiting for a dependent-launch
// dependency that never resolves (observed as 98% SM util at idle power with
// zero memory traffic). Gates the PDL launch attribute off on SM110.
#pragma once

#include <cuda_runtime.h>

namespace vllm_stable {

inline bool disable_pdl_sm110() {
  static const bool value = []() {
    int32_t major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    return major * 10 + minor == 110;
  }();
  return value;
}

}  // namespace vllm_stable
