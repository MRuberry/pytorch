#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/fusers/config.h"
#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/fusion_spec.h"
#include "torch/csrc/jit/fusers/return_code.h"

#include <cstdint>
#include <vector>

namespace torch { namespace jit { namespace fusers {

// Returns true if the op is fusable, false o.w.
bool isSupportedOp(const Node* node);

// Performs device-independent compilation of the given fusion_group
// and returns a (int64_t) key that can be used to run it 
// (see runFusion below) in the future.
ReturnCode registerFusion(int64_t& key, const Node* fusion_group);

//
// bool compileFusion(
//   const FusionSpec& spec
// , const FusionArgSpec& arg_spec
// , const std::vector<int64_t>& map_size);

} // namespace fusers
} // namespace jit
} // namespace torch
