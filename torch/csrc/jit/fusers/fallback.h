#pragma once

#include "torch/csrc/jit/stack.h"

#include <cstdlib>

namespace torch { namespace jit { namespace fusers {

void runFallback(int64_t key, Stack& stack);

} // namespace fusers
} // namespace jit
} // namespace torch
