#pragma once

#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/fusers/return_code.h"

#include <cstdint>

namespace torch { namespace jit { namespace fusers {

// Runs the fusion associated with the key (from registerFusion above)
// on the inputs taken from the given Stack.
ReturnCode runFusion(
  const int64_t key
, Stack& stack);

} // namespace fusers
} // namespace jit
} // namespace torch
