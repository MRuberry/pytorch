#pragma once
#include "torch/csrc/jit/fusers/config.h"
#if USE_CUDA_FUSER

#include <string>
#include <cstdint>

namespace torch { namespace jit { namespace fusers { namespace cuda {

bool compileCUDA(
  const int64_t device
, const std::string& name
, const std::string& code);

} // namespace cuda
} // namespace fusers
} // namespace jit
} // namespace torch

#endif // USE_CUDA_FUSER
