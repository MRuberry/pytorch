#include <torch/csrc/jit/fuser/common/utils.h>

#include <c10/util/Exception.h>

#include <algorithm>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

c10::Device getFusionDevice(const Node* const fusion) {
  const std::shared_ptr<c10::TensorType> out_tensor = fusion->outputs()[0]->type()->expect<TensorType>();
  return *(out_tensor->device());
}

c10::DeviceType getFusionDeviceType(const Node* const node) {
  return getFusionDevice(node).type();
}

size_t getRank(const std::shared_ptr<c10::TensorType>& tensor) {
  return *(tensor->dim());
}

size_t getNumel(const std::shared_ptr<c10::TensorType>& tensor) {
  return *(tensor->numel());
}

size_t getNumNonCollapsibleDims(const std::shared_ptr<c10::TensorType>& tensor) {
  const c10::VaryingShape& sizes = tensor->sizes();
  const c10::VaryingStrides& strides = tensor->strides();

  const auto nDims = getRank(tensor);

  if (nDims == 0) {
    return 0;
  }

  // Finds last dim with size > 1
  auto last = nDims - 1;
  for (int i = static_cast<int>(last); i >=0; --i) {
    const auto size = *(sizes[i]);
    if (size == 0) {
      return 0;
    } else if (size == 1) {
      continue;
    } else {
      last = i;
      break;
    }
  }

  size_t nNonCollapsibleDims = 1;
  auto collapse_value = *(strides[last]) * *(sizes[last]);
  for (int i = static_cast<int>(last - 1); i >= 0; --i) {
    const auto stride = *(strides[i]);
    const auto size = *(sizes[i]);

    // Tensors with a size of zero are empty
    // Size 1 dims are always collapsible
    if (size == 0) {
      return 0;
    } else if (size == 1) {
      continue;
    }

    if (stride != collapse_value) {
      ++nNonCollapsibleDims;
    }

    collapse_value = size * stride;
  }

  return nNonCollapsibleDims;
}

float getAsFloat(const Value* const value) {
 if (value->type() == FloatType::get()) {
   return value->node()->f(attr::value);
 }
 if (value->type() == IntType::get()) {
   return static_cast<float>(value->node()->i(attr::value));
 }

 TORCH_CHECK(false, "getAsFloat() found unknown scalar type!");
}

c10::optional<float> getFloat(const Value* const value) {
  if (value->type() == FloatType::get()) {
    return value->node()->f(attr::value);
  }

  return c10::nullopt;
}

c10::optional<int> getInt(const Value* const value) {
  if (value->type() == IntType::get()) {
    return value->node()->i(attr::value);
  }

  return c10::nullopt;
}

bool haveSameDevice(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
) {
  const auto lhs_device = *(lhs->device());
  const auto rhs_device = *(rhs->device());
  return (lhs_device == rhs_device);
}

bool haveSameScalarType(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
) {
  const auto lhs_scalar_type = *(lhs->scalarType());
  const auto rhs_scalar_type = *(rhs->scalarType());
  return (lhs_scalar_type == rhs_scalar_type);
}

bool haveSameSizes(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
) {
  const auto& lhs_sizes = lhs->sizes();
  const auto& rhs_sizes = rhs->sizes();

  if (*(lhs_sizes.size()) != *(rhs_sizes.size())) {
    return false;
  }

  for (size_t i = 0; i < *(lhs_sizes.size()); ++i) {
    if (*(lhs_sizes[i]) != *(rhs_sizes[i])) {
      return false;
    }
  }

  return true;
}

bool haveSameStrides(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
) {
  const auto& lhs_strides = lhs->strides();
  const auto& strides = rhs->strides();

  if (*(lhs_strides.size()) != *(strides.size())) {
    return false;
  }

  for (size_t i = 0; i < *(lhs_strides.size()); ++i) {
    if (*(lhs_strides[i]) != *(strides[i])) {
      return false;
    }
  }

  return true;
}

bool haveSameShape(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
) {
  return (
     haveSameDevice(lhs, rhs)
  && haveSameScalarType(lhs, rhs)
  && haveSameSizes(lhs, rhs)
  && haveSameStrides(lhs, rhs));
}

// TODO: get device
TensorMeta collapse(
  const std::shared_ptr<c10::TensorType>& out
, const std::shared_ptr<c10::TensorType>& tensor
) {
  // Checks that both tensors have the same rank
  const auto rank = getRank(out);
  TORCH_CHECK(getRank(tensor) == rank, "Trying to collapse a tensor with different rank than its output!");

  // Cases:
  // (1) The tensor has a dimension of size 0 - collapse is 0D
  // (2) The tensor is broadcast / discontiguous - collapse is 1 + X + Y
  //      where X is the number of broadcast dimensions (except the innermost)
  //      and Y is the number of distinct discontiguities
  // (3) The tensor has no discontiguities and does not broadcast - collapse is 1

  // Checks for 0D tensor (has a dimension with size = 0)
  const auto numel = getNumel(tensor);
  if (numel == 0) {
    return TensorMeta{c10::DeviceType::CPU};
  }

  // Checks for broadcast discontiguity
  // Broadcast occurs when this tensor has size 1 in a dim
  //  out tensor has size != 1.
  // Discontiguities occur when a dimension and the succeeding dimension
  //  have the following properties:
  //    (1) the next dimension is not broadcasting and has size 1
  //    (2) the next dimension size * stride == the stride of the current dimension

  const c10::VaryingShape& sizes = tensor->sizes();
  const c10::VaryingStrides& strides = tensor->strides();
  const c10::VaryingShape& out_sizes = out->sizes();

  std::vector<int64_t> collapsed_sizes;
  std::vector<int64_t> collapsed_strides;

  int64_t prior_size = -1;
  int64_t prior_stride = -1;

  auto hasPrior = [&prior_size]() {
    return (prior_size != -1);
  };

  auto emitPrior = [&prior_size, &prior_stride, &collapsed_sizes, &collapsed_strides]() {
    if (prior_size != -1) {
      collapsed_sizes.emplace_back(prior_size);
      collapsed_strides.emplace_back(prior_stride);
    }
    prior_size = -1;
    prior_stride = -1;
  };

  for (auto i = decltype(rank){0}; i < rank; ++i) {
    const auto size = *(sizes[i]);
    const auto stride = *(strides[i]);
    const auto out_size = *(out_sizes[i]);

    // Preservers broadcasts but skips non-broadcasting dimensions of size 1
    // Non-broadcasting dims of size 1 are skipped
    //  (unless they are the innermost dim and there are no other dims)
    if (size == 1) {
      if (out_size != 1) {
        emitPrior();
        collapsed_sizes.emplace_back(1);
        collapsed_strides.emplace_back(0);
      } else if (i == rank - 1) {
        if (hasPrior()) {
          emitPrior();
        } else {
          collapsed_sizes.emplace_back(1);
          collapsed_strides.emplace_back(0);
        }
      }
      continue;
    }

    // If contiguous with prior dimension, merges the prior dim
    // Otherwise, emits the prior (possibly merged) dim(s) and updates
    // the tracked size and stride.
    if (hasPrior() && (size * stride == prior_stride)) {
      prior_size *= size;
      prior_stride = stride;
    } else {
      emitPrior();
      prior_size = size;
      prior_stride = stride;
    }

    // Checks for end of array
    if (i == rank - 1) {
      emitPrior();
    }
  }

  return TensorMeta{
    c10::DeviceType::CPU
  , std::move(collapsed_sizes)
  , std::move(collapsed_strides)};
}

void printMeta(const TensorMeta& meta) {
  std::cout << "TensorMeta{" << meta.device_type_ << ", ";

  // print sizes
  std::cout << "sizes: [";
  const auto& sizes = meta.sizes();
  for (auto i = decltype(meta.rank()){0}; i < meta.rank(); ++i) {
    std::cout << sizes[i];

    if (i != meta.rank() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "], ";

  // print strides
  std::cout << "strides: [";
  const auto& strides = meta.strides();
  for (auto i = decltype(meta.rank()){0}; i < meta.rank(); ++i) {
    std::cout << strides[i];

    if (i != meta.rank() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]";

  std::cout << "}" << std::endl;
}

}}} // namespace torch::jit::fuser
