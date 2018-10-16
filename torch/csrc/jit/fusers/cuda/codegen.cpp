#include "torch/csrc/jit/fusers/cuda/codegen.h"

#include "torch/csrc/jit/fusers/cuda/resource_strings.h"

namespace torch { namespace jit { namespace fusers { namespace cuda {

bool generateCUDA(
  std::stringstream& ss
, TemplateEnv& env
, const bool has_half_tensor
, const bool has_random) {

  if (has_half_tensor) {
    env.s("HalfHeader", half_support_literal);
  } else {
    env.s("HalfHeader", "");
  }

  if (has_random) {
    env.s("RandHeader", cuda::rand_support_literal);
    env.s("RandParam", cuda::rand_param);
    env.s("RandInit", cuda::rand_init);
  } else {
    env.s("RandHeader", "");
    env.s("RandParam", "");
    env.s("RandInit", "");
  }

  env.s("type_declarations", type_declarations_template.format(env));
  ss << cuda_compilation_unit_template.format(env);

  return true;
}

} // namespace cuda
} // namespace fusers
} // namespace jit 
} // namespace torch
