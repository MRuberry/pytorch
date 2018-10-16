#pragma once
#include "torch/csrc/jit/fusers/config.h"
#if USE_CUDA_FUSER

#include "torch/csrc/jit/code_template.h"

#include <sstream>

namespace torch { namespace jit { namespace fusers { namespace cuda {

bool generateCUDA(
  std::stringstream& ss
, TemplateEnv& env
, const bool has_half_tensor
, const bool has_random);

} // namespace cuda
} // namespace fusers
} // namespace jit
} // namespace torch

#endif // USE_CUDA_FUSER
