#pragma once
#include "torch/csrc/jit/fusers/config.h"
#if USE_CUDA_FUSER

#include "torch/csrc/jit/fusers/interface.h"

#include "torch/csrc/jit/ir.h"

#include "ATen/ATen.h"

#include <vector>
#include <memory>

namespace torch { namespace jit { namespace cudafuser {

// Returns a cached fusion or compiles and caches a new fusion for the 
// given group.
std::shared_ptr<FusionHandle> getFusionHandle(Node* fusion_group);

// Turns the graph into a fusion group, gets a fusion for it,
// and runs that fusion using the given inputs. Returns the output(s).
std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs);

} // namespace cudafuser
} // namespace jit 
} // namespace torch

#endif // USE_CUDA_FUSER
