#pragma once

#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/fusers/fusion_spec.h"
#include "torch/csrc/jit/fusers/arg_info.h"
#include "torch/csrc/jit/fusers/return_code.h"

namespace torch { namespace jit { namespace fusers {

// Fills in arg_info from the stack inputs and the operations 
// recorded in spec. 
// ArgInfo describes input, intermediate, and output arguments, 
// including their names, scalar types, number of elements, and shape.
// Returns a return code signifying success or a failure state.
ReturnCode typeAndShapePropagation(
  ArgInfo& arg_info
, const FusionSpec& spec
, const Stack& stack
);

} // namespace fusers
} // namespace jit
} // namespace torch
