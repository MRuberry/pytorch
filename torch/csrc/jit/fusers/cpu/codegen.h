#pragma once
#include "torch/csrc/jit/fusers/config.h"
#if USE_CPU_FUSER

#include "torch/csrc/jit/code_template.h"

#include <sstream>

namespace torch { namespace jit { namespace fusers { namespace cpu {

bool generateCpp(
  std::stringstream& ss
, TemplateEnv& env
, const bool has_half_tensor
, const bool has_random);

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch

#endif // USE_CPU_FUSER
