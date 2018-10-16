#pragma once
#include "torch/csrc/jit/fusers/config.h"
#if USE_CPU_FUSER || USE_CUDA_FUSER

#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/fusers/tensor_desc.h"

#include <memory>
#include <cstdint>
#include <vector>

namespace torch { namespace jit { namespace fusers {

// Descriptor for chunk-ing an input tensor into subtensors
// OR concat-ing an output tensor from subtensors
struct PartitionDesc {
  
  PartitionDesc()
  : nSubTensors{1}
  , dim{0} { }

  PartitionDesc(
    const TensorDesc& _desc
  , size_t _nSubTensors
  , size_t _dim)
  : nSubTensors{_nSubTensors}
  , dim{_dim} {
    JIT_ASSERT(nSubTensors > 1);
    std::vector<bool> cont = _desc.contiguity;
    if(dim > 0) {
      // when we narrow the concatenated output/chunked input
      // we make the size[dim] smaller while keeping the stride[dim] the same,
      // meaning: stride[dim - 1] != stride[dim]*size[dim]
      // so dim - 1 is no longer contiguous
      cont[dim - 1] = false;
    }
    subTensorDesc.reset(new TensorDesc(_desc.scalar_type, cont));
  }

  bool isNoop() const {
    return nSubTensors == 1;
  }

  size_t nSubTensors; // == 1 for tensors that should not be operated on via chunk/cat
  size_t dim; // dimension along which the chunk/concat occurs
  std::unique_ptr<TensorDesc> subTensorDesc; // descriptor for the subtensor, if it exists
};

} // namespace fusers
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
