#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit { namespace fusers {

bool isSimpleMap(const Node* n);

} // namespace fusers
} // namespace jit
} // namespace torch
