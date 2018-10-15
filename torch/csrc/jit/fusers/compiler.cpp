#include "torch/csrc/jit/fusers/compiler.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/fusion_spec.h"
#include "torch/csrc/jit/fusers/cache.h"

#include <iostream>
#include <vector>
#include <memory>
#include <unordered_set>

namespace torch { namespace jit { namespace fusers {

static Node* usedInFusedChunk(Value* input) {
  auto uses = input->uses();
  if (uses.size() == 1) {
    Node *user = uses[0].user;
    if (user->kind() == prim::ConstantChunk) {
      return user;
    }
  }
  return nullptr;
}

static std::vector<PartitionInfo> getInputChunkDescriptors(std::shared_ptr<Graph> graph) {
  std::vector<PartitionInfo> descs;
  descs.reserve(graph->inputs().size());
  for (Value* input : graph->inputs()) {
    if (Node* chunk = usedInFusedChunk(input)) {
      descs.emplace_back(chunk->i(attr::chunks), chunk->i(attr::dim));
    } else {
      descs.emplace_back(1, 0);
    }
  }
  return descs;
}

static std::vector<int64_t> getInputDependencies(Value* output) {
  // Run a DFS traversal to find all inputs that affect a given output value
  std::vector<Value*> queue{output};
  std::unordered_set<Value*> inputs;
  std::unordered_set<Value*> seen;
  while (!queue.empty()) {
    Value* val = queue.back(); queue.pop_back();
    Node* producer = val->node();
    if (producer->kind() == prim::Param) {
      inputs.insert(val);
      continue;
    }
    for (Value* input : producer->inputs()) {
      if (/*bool inserted = */seen.insert(input).second) {
        queue.push_back(input);
      }
    }
  }

  // Convert Value* into offsets into the graph's input list
  std::vector<int64_t> offsets;
  offsets.reserve(inputs.size());
  for (Value* input : inputs) {
    offsets.push_back(input->offset());
  }

  std::sort(offsets.begin(), offsets.end());
  return offsets;
}

static std::vector<std::vector<int64_t>> getInputBroadcastGroups(std::shared_ptr<Graph> graph) {
  std::unordered_set<std::vector<int64_t>, torch::hash<std::vector<int64_t>>> broadcast_groups;
  for (Value* output : graph->outputs()) {
    broadcast_groups.insert(getInputDependencies(output));
  }
  return std::vector<std::vector<int64_t>>{broadcast_groups.begin(), broadcast_groups.end()};
}

int64_t registerFusion(Node* fusion_group) {
  // Creates FusionSpec
  auto graph = fusion_group->g(attr::Subgraph)->copy();
  EraseShapeInformation(*graph);
  const auto key = store(graph);
  const auto spec = retrieve(key);

  if (!spec) {
    // TODO: error out
  }
  
  // Performs device independent compilation of the spec 
  if (canFuseOnCPU() || canFuseOnGPU()) {
    const auto broadcast_groups = getInputBroadcastGroups(graph);
    const auto chunk_descs = getInputChunkDescriptors(graph);
    (*spec).setInputBroadcastGroups(broadcast_groups);
    (*spec).setinputChunkDescriptors(chunk_descs);
  }

  return key;
}

bool runFusion(
  const int64_t key
, Stack& stack) {
  if (!canFuseOnCPU() && !canFuseOnGPU()) {
    return false;
  }

  // TODO: impl
  return false;
}

} // namespace fusers
} // namespace jit
} // namespace torch
