#include "torch/csrc/jit/fusers/cuda/interface.h"

#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/common/fusion_spec.h"
#include "torch/csrc/jit/fusers/common/fusion_handle_impl.h"


#include "torch/csrc/utils/functional.h" //fmap
#include "torch/csrc/jit/ivalue.h" // IValue

#include "torch/csrc/jit/assertions.h"

#include <string>
#include <tuple>
#include <cstdlib>

namespace torch { namespace jit { namespace cudafuser {

static std::unordered_map<std::string, std::shared_ptr<FusionHandleImpl>> cache_map;

std::shared_ptr<FusionHandle> getFusionHandle(Node* node) {
  // Short-circuits if not on GPU
  const auto device = node->i(attr::device);
  JIT_ASSERT(device != kCPUDevice);

  // Canonicalizes graph 
  auto graph = node->g(attr::Subgraph)->copy();
  EraseShapeInformation(*graph);

  // Creates key
  std::stringstream key;
  key << "device " << device << "\n";
  key << *graph << "\n";
  std::string key_ = key.str();

  // Looks for cached fusion, creates a new one if not found
  auto it = cache_map.find(key_);
  if (it == cache_map.end()) {
    std::tie(it, std::ignore) = 
      cache_map.emplace(
        key_
      , std::make_shared<FusionHandleImpl>(graph, device));
  }

  return it->second;
}

std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs) {
  // Creates a graph consisting of a single fusion group
  // whose contents are a copy of the given graph
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group = 
    wrapper_graph->insertNode(wrapper_graph->createFusionGroup(device));
  fusion_group->g_(attr::Subgraph, graph.copy());
  
  // Updates input and output metadata
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // Acquires and runs the handle, returning the outputs 
  auto handle = cudafuser::getFusionHandle(fusion_group);
  Stack stack = fmap<IValue>(inputs);
  handle->run(stack);
  return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}

} // namespace cudafuser
} // namespace jit 
} // namespace torch
