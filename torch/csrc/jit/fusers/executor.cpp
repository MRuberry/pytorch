#include "torch/csrc/jit/fusers/executor.h"

#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "ATen/core/optional.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/cache.h"
#include "torch/csrc/jit/fusers/fusion_spec.h"
#include "torch/csrc/jit/fusers/compiler.h"
#include "torch/csrc/jit/fusers/arg_info.h"
#include "torch/csrc/jit/fusers/type_and_shape_propagation.h"

#include <vector>
#include <tuple>
#include <stdexcept>
#include <algorithm>
#include <map>
#include <iostream> // TODO: remove me, debugging only

namespace torch { namespace jit { namespace fusers {

ReturnCode runFusion(
  const int64_t key
, Stack& stack) {
  // Short-circuits if fusion isn't enabled
  if (!canFuseOnCPU() && !canFuseOnGPU()) return ReturnCode::FUSION_DISABLED;

  // Acquires the FusionSpec
  auto maybe_spec = retrieve(key);
  if (!maybe_spec) 
    throw std::runtime_error("Failed to find fusion specification to run");
  auto& spec = *maybe_spec;

  // Short-circuits if the spec isn't fusable
  if (!spec.isFusable()) return ReturnCode::NONFUSABLE_SPEC;

  // Performs type and shape propagation, short-circuiting on failures
  ArgInfo arg_info;
  const auto prop_result = typeAndShapePropagation(arg_info, spec, stack);
  // std::map<int64_t, TensorDesc> tensor_map;
  // auto prop_result = typeAndShapePropagation(tensor_map, spec, stack);
  // const auto prop_rc = std::get<0>(prop_result);
  // if (prop_rc != ReturnCode::SUCCESS) return prop_rc;
  // auto arg_spec = *(std::get<1>(prop_result));

  // Looks for existing compilation
  // TODO
  // auto it = kernels.find(spec);
  // if (it == kernels.end()) {
  //   std::tie(it, std::ignore) = kernels.emplace(spec, compileSpec(spec, *maybe_map_size));
  // }
  // auto& fn = it->second;

  // Performs runtime compilation
  // auto map_size = *(std::get<2>(prop_result));
  // const auto comp_result = compileFusion(spec, arg_spec, map_size); 

  // Acquires compilation result

  // std::vector<at::Tensor> outputs;
  // fn->launch(args, outputs);
  // drop(stack, num_inputs);
  // stack.insert(
  //   stack.end()
  // , std::make_move_iterator(outputs.begin())
  // , std::make_move_iterator(outputs.end()));

  // TODO: impl
  return ReturnCode::NOT_IMPL;
}

} // namespace fusers
} // namespace jit
} // namespace torch
