#include "torch/csrc/jit/fusers/interface.h"

#include "torch/csrc/jit/fusers/config.h"
#include "torch/csrc/jit/fusers/common/cache.h"

#if USE_CPU_FUSER
  #include "torch/csrc/jit/fusers/cpu/interface.h"
#endif // USE_CPU_FUSER

#if USE_CUDA_FUSER
  #include "torch/csrc/jit/fusers/cuda/interface.h"
#endif // USE_CUDA_FUSER

#include <stdexcept>

namespace torch { namespace jit {

namespace detail {

bool cpu_fuser_enabled = true; // TODO: disable (only enabled for testing)

} // namespace detail

// Returns true if the fusion_group was unregistered, false otherwise.
const std::string& registerFusion(Node* fusion_group) {
  if (fusion_group->hasAttribute(attr::fusion_key)) {
    return fusion_group->s(attr::fusion_key);
  }

  // Stores the fusion group, stuffs and returns the key
  auto& cache = fusers::getCache();
  auto& key = cache.storeOnce(fusion_group);
  fusion_group->s_(attr::fusion_key, key);
  return key;
}

// Returns true if the fusion was run, false if a fallback was run.
bool runFusion(const std::string& key, Stack& stack) {
  std::cout << std::endl << "fusers/runFusion" << std::endl;
  // Acquires the FusionSpec from the global cache (by key)
  auto& cache = fusers::getCache();
  auto spec = cache.get(key);
  if (!spec) {
    // TODO: assert
    std::cout << std::endl << "No spec!" << std::endl;
    return false;
  }

  // Selects an appropriate device fuser
  // Asks the fuser if they can run the spec with the given inputs
  // If not, runs the fallback
  // Otherwise, requests the fuser run the spec with the inputs

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
      return fusers::cpu::debugLaunchGraph(graph, inputs);
    #endif // USE_CPU_FUSER
    throw std::runtime_error("CPU fusion is not supported on this build.");
  }

  #if USE_CUDA_FUSER
    return fusers::cuda::debugLaunchGraph(graph, device, inputs);
  #endif // USE_CUDA_FUSER

  throw std::runtime_error("CUDA fusion is not supported on this build.");
}

} // namespace jit
} // namespace torch
