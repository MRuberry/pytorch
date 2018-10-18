#include "torch/csrc/jit/fusers/interface.h"

#include "torch/csrc/jit/fusers/config.h"
#include "torch/csrc/jit/fusers/compiler.h"
#include "torch/csrc/jit/fusers/fallback.h"
#include "torch/csrc/jit/fusers/executor.h"
#include "torch/csrc/jit/fusers/simple_mappable.h"
#include "torch/csrc/jit/fusers/return_code.h"

#include <stdexcept>
#include <iostream> // TODO: remove me

namespace torch { namespace jit {

namespace detail {

bool cpu_fuser_enabled = true; // TODO: reset to false, true for debug only!

} // namespace detail

bool isSupportedOp(const Node* node) {
  return fusers::isSupportedOp(node);
}

int64_t registerFusion(const Node* fusion_group) {
  int64_t key;
  const auto result = fusers::registerFusion(key, fusion_group);
  if (result != fusers::ReturnCode::SUCCESS) {
    // TODO: remove me
    std::cout << result << std::endl;
  }

  return key;
}

bool runFusion(
  const int64_t key
, Stack& stack) {
  const auto result = fusers::runFusion(key, stack);
  if (result != fusers::ReturnCode::SUCCESS) {
    // TODO: remove me
    std::cout << result << std::endl;
    return false;
  }

  return true;
}

void runFallback(const int64_t key, Stack& stack) {
  fusers::runFallback(key, stack);
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
