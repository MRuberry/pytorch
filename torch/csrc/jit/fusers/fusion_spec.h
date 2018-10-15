#pragma once
#include "torch/csrc/jit/fusers/config.h"
#if USE_CPU_FUSER || USE_CUDA_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/common/fusion_arg_spec.h"
#include "torch/csrc/jit/fusers/common/fused_kernel.h"

#include <memory>
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace torch { namespace jit { namespace fusers {

struct PartitionInfo {
  PartitionInfo(
    const int64_t _nSubTensors
  , const int64_t _dim)
  : nSubTensors_{_nSubTensors}, dim_{_dim} 
  { }

  int64_t nSubTensors() const { return nSubTensors_; }
  int64_t dim() const { return dim_; }

private:
  int64_t nSubTensors_;
  int64_t dim_;
};

// "Fusion Specification." - Contains device-independent fusion information.
struct FusionSpec {
  FusionSpec(
    const int64_t _key
  , std::shared_ptr<Graph> _graph)
  : key_{_key}
  , graph_{_graph} 
  , code_{_graph}
  { }

  int64_t key() const { return key_; }
  std::shared_ptr<Graph> graph() const { return graph_; }
  const Code& code() const { return code_; }
  
  // Getters and setters for inputBroadcastGroups_
  const std::vector<std::vector<int64_t>>& inputBroadcastGroups() const { return inputBroadcastGroups_; }
  std::vector<std::vector<int64_t>>&inputBroadcastGroups() { return inputBroadcastGroups_; }
  void setInputBroadcastGroups(std::vector<std::vector<int64_t>> _inputBroadcastGroups) { 
    inputBroadcastGroups_ = _inputBroadcastGroups;
  }

  // Getters and setters for inputChunkDescriptors_
  const std::vector<PartitionInfo>& inputChunkDescriptors() const { return inputChunkDescriptors_; }
  std::vector<PartitionInfo>& inputChunkDescriptors() { return inputChunkDescriptors_; }
  void setinputChunkDescriptors( std::vector<PartitionInfo> _inputChunkDescriptors) { 
    inputChunkDescriptors_ = _inputChunkDescriptors;
  }

private:  
  int64_t key_;
  std::shared_ptr<Graph> graph_;
  Code code_;
  std::vector<std::vector<int64_t>> inputBroadcastGroups_;
  std::vector<PartitionInfo> inputChunkDescriptors_;
};

} // namespace fusers
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
