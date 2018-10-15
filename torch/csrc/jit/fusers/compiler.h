#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/fusers/config.h"
#include "torch/csrc/jit/fusers/interface.h"

#include <cstdint>

namespace torch { namespace jit { namespace fusers {

// Performs device-independent compilation of the given fusion_group
// and returns a (int64_t) key that can be used to run it 
// (see runFusion below) in the future.
int64_t registerFusion(Node* fusion_group);

// Runs the fusion associated with the key (from registerFusion above)
// on the inputs taken from the given Stack.
bool runFusion(
  const int64_t key
, Stack& stack);

} // namespace fusers
} // namespace jit
} // namespace torch
