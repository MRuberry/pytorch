#pragma once
#include "torch/csrc/jit/fusers/config.h"
#if USE_CUDA_FUSER

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/fusers/interface.h"

#include "ATen/ATen.h"

#include <vector>
#include <memory>

namespace torch { namespace jit { namespace fusers { namespace cuda {

// Runs the fusion specified by the key
// Returns true if the fusion was run, false if a fallback was run
bool runFusion(const std::string& key, Stack& stack);

// Turns the graph into a fusion group, gets a fusion for it,
// and runs that fusion using the given inputs. Returns the output(s).
std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs);

} // namespace cuda
} // namespace fusers
} // namespace jit 
} // namespace torch

#endif // USE_CUDA_FUSER
