#include "torch/csrc/jit/fusers/cpu/interface.h"

#include "torch/csrc/utils/functional.h" //fmap
#include "torch/csrc/jit/ivalue.h" // IValue
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/passes/shape_analysis.h" // EraseShapeInformation
#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/common/cache.h"
#include "torch/csrc/jit/fusers/common/fusion_handle_impl.h"

#include <cstdlib>
#include <string>
#include <sstream>
#include <tuple>

namespace torch { namespace jit { namespace fusers { namespace cpu {

/* Compiler Config */

static const std::string check_exists_string = "which '${program}' > /dev/null";

static bool programExists(const std::string& program) {
  TemplateEnv env;
  env.s("program", program);
  std::string cmd = format(check_exists_string, env);
  return 0 == system(cmd.c_str());
}

CompilerConfig::CompilerConfig() {
  const char* cxx_env = getenv("CXX");
  if (cxx_env != nullptr) {
    cxx = cxx_env;
  }

  if (!programExists(cxx)) {
    cxx = "";
  }
  
  const char* debug_env = getenv("PYTORCH_FUSION_DEBUG");
  debug = debug_env && atoi(debug_env) != 0;
}

CompilerConfig& getCompilerConfig() {
  static CompilerConfig config;
  return config;
}


// std::shared_ptr<FusionHandle> CPUFusionCompiler::getFusionHandle(Node* fusion_group) {
//   int device = fusion_group->i(attr::device);
//   JIT_ASSERT(device == kCPUDevice);
//   auto graph = fusion_group->g(attr::Subgraph)->copy();
//   EraseShapeInformation(*graph);
//   std::stringstream key;
//   key << "device " << device << "\n";
//   key << *graph << "\n";
//   std::string key_ = key.str();
//   auto it = cache_map.find(key_);
//   if (it == cache_map.end()) {
//     std::tie(it, std::ignore) = cache_map.emplace(key_, std::make_shared<FusionHandleImpl>(graph, device));
//   }
//   return it->second;
// }

bool runFusion(const std::string& key, Stack& stack) {
  std::cout << std::endl << "cpu/runFusion" << std::endl;

  return false;
}

std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, at::ArrayRef<at::Tensor> inputs) {
  return {};
  // auto wrapper_graph = std::make_shared<Graph>();
  // Node* fusion_group = wrapper_graph->insertNode(wrapper_graph->createFusionGroup(kCPUDevice));
  // fusion_group->g_(attr::Subgraph, graph.copy());
  // for (size_t i = 0; i < graph.inputs().size(); ++i) {
  //   fusion_group->addInput(wrapper_graph->addInput());
  // }
  // for (size_t i = 0; i < graph.outputs().size(); ++i) {
  //   wrapper_graph->registerOutput(fusion_group->addOutput());
  // }
  // auto cache = getFusionHandle(fusion_group);
  // Stack stack = fmap<IValue>(inputs);
  // cache->run(stack);
  // return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}


} // namespace cpu
} // namespace fusers
} // namespace jit
} // namespace torch
