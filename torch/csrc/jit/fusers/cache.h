#pragma once

#include "ATen/core/optional.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fusers/fusion_spec.h"

#include <cstdint> 

namespace torch { namespace jit { namespace fusers {

int64_t store(std::shared_ptr<Graph> graph);

at::optional<FusionSpec&> retrieve(const int64_t key);

} // namespace fusers
} // namespace jit
} // namespace torch
