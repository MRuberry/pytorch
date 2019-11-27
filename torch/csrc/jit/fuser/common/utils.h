#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h> // TORCH_API
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>

#include <aten/src/ATen/core/jit_type.h>
#include <torch/csrc/jit/ir.h>

#include <torch/csrc/jit/fuser/common/tensor_meta.h>

namespace torch {
namespace jit {
namespace fuser {

// Warning: assumes all fusion outputs are complete tensors
TORCH_API c10::Device getFusionDevice(const Node* const node);

TORCH_API c10::DeviceType getFusionDeviceType(const Node* const node);

TORCH_API size_t getRank(const std::shared_ptr<c10::TensorType>& tensor);

TORCH_API size_t getNumel(const std::shared_ptr<c10::TensorType>& tensor);

TORCH_API size_t getNumNonCollapsibleDims(const std::shared_ptr<c10::TensorType>& tensor);

TORCH_API float getAsFloat(const ::torch::jit::Value* const value);

TORCH_API c10::optional<float> getFloat(const Value* const value);

TORCH_API c10::optional<int> getInt(const Value* const value);

TORCH_API bool haveSameDevice(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
);

TORCH_API bool haveSameScalarType(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
);

TORCH_API bool haveSameSizes(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
);

TORCH_API bool haveSameStrides(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
);

TORCH_API bool haveSameShape(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
);

TORCH_API TensorMeta collapse(
  const std::shared_ptr<c10::TensorType>& out
, const std::shared_ptr<c10::TensorType>& tensor
);

TORCH_API void printMeta(
  const TensorMeta& meta
);



}}} // namespace torch::jit::fuser
