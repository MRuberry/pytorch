#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"

namespace torch { namespace jit { namespace fusers {

void runFallback(Node* fusion_group, Stack& stack);

} // namespace fusers
} // namespace jit
} // namespace torch
