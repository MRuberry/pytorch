#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"

#include "torch/csrc/WindowsTorchApiMacro.h"

#include "ATen/ATen.h"

#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace torch { namespace jit {

constexpr int kUndefinedDevice = -2;
constexpr int kCPUDevice = -1;

// Creates a fusion plan for the given node, returning a std::string
// to act as a key to signify that fusion.
TORCH_API int64_t registerFusion(Node* fusion_group);

// Runs the fusion assigned to the given fusion_key (see registerFusion()) 
// using the inputs on the given Stack.
// Returns true if the fusion was run as expected and false otherwise.
TORCH_API bool runFusion(const int64_t key, Stack& stack);

TORCH_API void runFallback(const int64_t key, Stack& stack);

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
