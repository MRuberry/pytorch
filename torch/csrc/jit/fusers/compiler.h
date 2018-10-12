#pragma once

#include "torch/csrc/jit/fusers/config.h"
#include "torch/csrc/jit/fusers/interface.h"

namespace torch { namespace jit { namespace fusers {

std::string registerFusion(Node* fusion_group);

bool runFusion(
  const std::string& fusion_key
, Stack& stack);

} // namespace fusers
} // namespace jit
} // namespace torch
