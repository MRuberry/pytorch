#include "torch/csrc/jit/fusers/interface.h"

#include "torch/csrc/jit/fusers/config.h"

#if USE_CPU_FUSER
  #include "torch/csrc/jit/fusers/cpu/interface.h"
#endif // USE_CPU_FUSER

#if USE_CUDA_FUSER
  #include "torch/csrc/jit/fusers/cuda/interface.h"
#endif // USE_CUDA_FUSER

#include <stdexcept>

namespace torch { namespace jit {

namespace detail {

bool cpu_fuser_enabled = false;

} // namespace detail

// Returns true if the fusion_group was unregistered, false otherwise.
bool registerFusion(Node* fusion_group) {
  return false;
}

// Returns true if the fusion was run, false if a fallback was run.
bool runFusion(Node* fusion_group, Stack& stack) {
  return false;
}

// TODO: remove me when fusion handle is removed
FusionHandle::~FusionHandle() { }

// std::shared_ptr<FusionHandle> getFusionHandle(Node* fusion_group) {
//   const auto device = fusion_group->i(attr::device);
//   if (device == kCPUDevice) {
//     #if USE_CPU_FUSER
//       return cpufuser::getFusionHandle(fusion_group);
//     #endif
//     throw std::runtime_error("CPU fusion is not supported on this build.");
//   }

//   #if USE_CUDA_FUSER
//     return cudafuser::getFusionHandle(fusion_group);
//   #endif // USE_CUDA_FUSER

//   throw std::runtime_error("CUDA fusion is not supported on this build.");
// }

bool canFuseOnCPU() {
  #if USE_CPU_FUSER
    return detail::cpu_fuser_enabled;
  #else // !USE_CPU_FUSER
    return false;
  #endif // USE_CPU_FUSER
}

bool canFuseOnGPU() {
  #if USE_CUDA_FUSER
    return true;
  #else // !USE_CUDA_FUSER
    return false;
  #endif // USE_CUDA_FUSER
}

void overrideCanFuseOnCPU(bool value) {
  detail::cpu_fuser_enabled = value;
}

std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs) {
  if (device == kCPUDevice) {
    #if USE_CPU_FUSER
      return cpufuser::debugLaunchGraph(graph, device, inputs);
    #endif // USE_CPU_FUSER
    throw std::runtime_error("CPU fusion is not supported on this build.");
  }

  #if USE_CUDA_FUSER
    return cudafuser::debugLaunchGraph(graph, device, inputs);
  #endif // USE_CUDA_FUSER

  throw std::runtime_error("CUDA fusion is not supported on this build.");
}

} // namespace jit
} // namespace torch
