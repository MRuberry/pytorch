#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h> // TORCH_API
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

TORCH_API bool isFusibleOnCPU(const Node* const node);

// TORCH_API int tryCreateFusion(const Node* const node);

// TORCH_API void compileFusion(const Node* const fusion);

// TORCH_API void callFusion(const int key, Stack& stack);

}}}} // namespace torch::jit::fuser::cpu
