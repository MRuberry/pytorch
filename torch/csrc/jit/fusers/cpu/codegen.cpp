#include "torch/csrc/jit/fusers/cpu/codegen.h"

#include "torch/csrc/jit/fusers/cpu/resource_strings.h"

namespace torch { namespace jit { namespace fusers { namespace cpu {

bool generateCpp(
  std::stringstream& ss
, TemplateEnv& env
, const bool has_half_tensor
, const bool has_random) {

  if (has_half_tensor || has_random) return false;

  env.s("type_declarations", type_declarations_template.format(env));
  ss << cpu_compilation_unit_template.format(env);

  return true;
}

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
