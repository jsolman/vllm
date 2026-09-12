// SPDX-License-Identifier: Apache-2.0
// PDL (programmatic dependent launch) policy for SM110 (Thor/Tegra, 20 SMs):
//  - Benefit is marginal at 20 SMs: the launch-overlap window PDL exploits is
//    small, so the upside hardly shows in end-to-end throughput.
//  - PDL is NOT known-broken on SM110. It was disabled by default out of
//    caution after observing instability (98% SM util at idle power with zero
//    memory traffic) while other bugs were also present; PDL was never
//    isolated as the cause.
//  - Set VLLM_SM110_PDL=1 to force-enable PDL launches for A/B testing.
#pragma once

#include <cuda_runtime.h>
#include <cstdlib>

namespace vllm_stable {

inline bool disable_pdl_sm110() {
  static const bool value = []() {
    const char* env = std::getenv("VLLM_SM110_PDL");
    if (env && env[0] == '1') return false;  // forced on for A/B
    int32_t major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    return major * 10 + minor == 110;
  }();
  return value;
}

}  // namespace vllm_stable
