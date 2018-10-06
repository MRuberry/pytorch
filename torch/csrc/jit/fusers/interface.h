#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"

#include "torch/csrc/WindowsTorchApiMacro.h"

#include "ATen/ATen.h"

#include <memory>
#include <vector>
#include <string>

namespace torch { namespace jit {

constexpr int kCPUDevice = -1;

// TODO: remove this, just for temporary compilation
struct TORCH_API FusionHandle {
  virtual void run(Stack& inputs) = 0;

  virtual ~FusionHandle() = 0;
};

// Returns true if the fusion_group was unregistered, false otherwise.
TORCH_API const std::string& registerFusion(Node* fusion_group);

// Returns true if the fusion was run, false if a fallback was run.
TORCH_API bool runFusion(const std::string& key, Stack& stack);

TORCH_API bool canFuseOnCPU();
TORCH_API bool canFuseOnGPU();

// CPU fuser is disabled by default, but we still want to test it.
TORCH_API void overrideCanFuseOnCPU(bool value);

TORCH_API std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs);

} // namespace jit
} // namespace torch
