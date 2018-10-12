#include "torch/csrc/jit/fusers/interface.h"

#include "torch/csrc/jit/fusers/config.h"
#include "torch/csrc/jit/fusers/compiler.h"
#include "torch/csrc/jit/fusers/fallback.h"

#include <stdexcept>

namespace torch { namespace jit {

namespace detail {

bool cpu_fuser_enabled = false;

} // namespace detail

std::string registerFusion(Node* fusion_group) {
  return fusers::registerFusion(fusion_group);
}

bool runFusion(
  const std::string& fusion_key
, Stack& stack) {
  return fusers::runFusion(fusion_key, stack);
}

void runFallback(Node* fusion_group, Stack& stack) {
  fusers::runFallback(fusion_group, stack);
}

bool canFuseOnCPU() {
  #if USE_CPU_FUSER
    return detail::cpu_fuser_enabled;
  #endif // USE_CPU_FUSER

  return false;
}

bool canFuseOnGPU() {
  #if USE_CUDA_FUSER
    return true;
  #endif  // USE_CUDA_FUSER

  return false;
}

void overrideCanFuseOnCPU(bool value) {
  detail::cpu_fuser_enabled = value;
}

std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs) {
  // if (device == kCPUDevice) {
  //   #if USE_CPU_FUSER
  //     return fusers::cpu::debugLaunchGraph(graph, device, inputs);
  //   #endif // USE_CPU_FUSER
  //   throw std::runtime_error("CPU fusion is not supported on this build.");
  // }

  // #if USE_CUDA_FUSER
  //   return fusers::cuda::debugLaunchGraph(graph, device, inputs);
  // #endif // USE_CUDA_FUSER
  return {};
  // throw std::runtime_error("CUDA fusion is not supported on this build.");
}

} // namespace jit
} // namespace torch
