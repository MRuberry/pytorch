#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"

#include "torch/csrc/WindowsTorchApiMacro.h"

#include "ATen/ATen.h"

#include <memory>
#include <vector>
#include <string>

namespace torch { namespace jit {

constexpr int kUndefinedDevice = -2;
constexpr int kCPUDevice = -1;

// Creates a fusion plan for the given node, returning a std::string
// to act as a key to signify that fusion.
TORCH_API std::string registerFusion(Node* fusion_group);

// Runs the fusion assigned to the given fusion_key (see registerFusion()) 
// using the inputs on the given Stack.
// Returns true if the fusion was run as expected and false otherwise.
TORCH_API bool runFusion(const std::string& fusion_key, Stack& stack);

TORCH_API void runFallback(Node* fusion_group, Stack& stack);

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
