#include "torch/csrc/jit/fusers/compiler.h"

namespace torch { namespace jit { namespace fusers {

std::string registerFusion(Node* fusion_group) {
  // Generates fusion key
  auto graph = fusion_group->g(attr::Subgraph)->copy();

  // EraseShapeInformation(*graph);
  // std::stringstream key;
  // key << "device " << device << "\n";
  // key << *graph << "\n";
  // std::string key_ = key.str();

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
