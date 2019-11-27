#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h> // TORCH_API
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>

#include <vector>
#include <cstdint>

namespace torch {
namespace jit {
namespace fuser {

struct TensorMeta {

TensorMeta(
  const c10::DeviceType _device_type)
: device_type_{_device_type} { }

TensorMeta(
  const c10::DeviceType _device_type
, std::vector<int64_t>&& _sizes
, std::vector<int64_t>&& _strides)
: device_type_{_device_type}
, sizes_{_sizes}
, strides_{_strides} { }

typename std::vector<int64_t>::size_type rank() const { return sizes_.size(); }

std::vector<int64_t>& sizes() { return sizes_; }
const std::vector<int64_t>& sizes() const { return sizes_; }

std::vector<int64_t>& strides() { return strides_; }
const std::vector<int64_t>& strides() const { return strides_; }

c10::DeviceType device_type_;
std::vector<int64_t> sizes_;
std::vector<int64_t> strides_;

};

}}} // namespace torch::jit::fuser
