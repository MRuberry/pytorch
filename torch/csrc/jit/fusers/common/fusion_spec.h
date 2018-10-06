#pragma once
#include "torch/csrc/jit/fusers/config.h"
#if USE_CPU_FUSER || USE_CUDA_FUSER

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/passes/shape_analysis.h" // EraseShapeInformation

#include <memory>
#include <sstream>
#include <string>

namespace torch { namespace jit { namespace fusers {

struct FusionSpec {
  FusionSpec() = default;
  ~FusionSpec() = default;

  FusionSpec(Node* node) {
    // Stores a canonicalized copy of the node's graph
    graph_ = node->g(attr::Subgraph)->copy();

    // Note: currently "canonicalization" is just removing shape information
    EraseShapeInformation(*graph_);
    
    // Creates key
    std::stringstream key;
    key << *graph_ << "\n";
    std::string key_ = key.str();
  }

  const std::shared_ptr<Graph> graph() { return graph_; }
  const std::string& key() { return key_; }

private:
  std::shared_ptr<Graph> graph_;
  std::string key_;
};

} // namespace fusers
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
