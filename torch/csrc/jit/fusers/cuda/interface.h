#pragma once
#include "torch/csrc/jit/fusers/config.h"
#if USE_CUDA_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/cuda/fusion_compiler.h"

#include <vector>
#include <memory>

namespace torch { namespace jit { namespace fusers { namespace cuda {

inline std::shared_ptr<FusionHandle> getFusionHandle(Node* fusion_group) {
  return getFusionCompiler().getFusionHandle(fusion_group);
}

std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs) {
  return getFusionCompiler().debugLaunchGraph(graph, device, inputs);
}

} // namespace cuda
} // namespace fusers
} // namespace jit 
} // namespace torch

#endif // USE_CUDA_FUSER
