#include "torch/csrc/jit/fusers/compiler.h"

#include "torch/csrc/jit/passes/shape_analysis.h"

#include <iostream>

namespace torch { namespace jit { namespace fusers {

std::string registerFusion(Node* fusion_group) {
  auto graph = fusion_group->g(attr::Subgraph)->copy();

  // Generates fusion key
  EraseShapeInformation(*graph);
  std::stringstream ss;
  ss << *graph;
  const auto key_ = ss.str();

  // auto it = cache_map.find(key_);
  // if (it == cache_map.end()) {
  //   std::tie(it, std::ignore) = 
  //     cache_map.emplace(
  //       key_
  //     , std::make_shared<FusionHandleImpl>(graph, device));
  // }

  // return it->second;

  return "0";
}

bool runFusion(
  const std::string& fusion_key
, Stack& stack) {
  if (!canFuseOnCPU && !canFuseOnGPU) return false;
  return false;
}

} // namespace fusers
} // namespace jit
} // namespace torch
