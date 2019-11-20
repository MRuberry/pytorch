#include <aten/src/ATen/core/jit_type.h>
#include <c10/core/DeviceType.h>

#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/cpu/interface.h>
#include <torch/csrc/jit/fuser/cuda/interface.h>
#include <torch/csrc/jit/fuser/common/utils.h>


#include <torch/csrc/jit/fuser/compiler.h>
#include <torch/csrc/jit/fuser/executor.h>
#include <torch/csrc/jit/fuser/fallback.h>
#include <torch/csrc/jit/fuser/kernel_cache.h>

#include <asmjit/asmjit.h>

#include <stdexcept>

namespace torch {
namespace jit {

namespace detail {

// Note: CPU fusion is currently disabled due to test flakiness
bool cpu_fuser_enabled = false;

} // namespace detail

// TODO: make dbgs take stream to print on
namespace {

void printScalar(const Value* const value) {
  if (value->node()->kind() == prim::Constant) {
    std::cout << "Const Scalar: ";
  } else {
    std::cout << "Scalar: ";
  }

  if (value->type() == FloatType::get()) {
    std::cout << "float ";
    const float val = value->node()->f(attr::value);
    std::cout << val;
  } else if (value->type() == IntType::get()) {
    std::cout << "int ";
    const int val = value->node()->i(attr::value);
    std::cout << val;
  } else {
    std::cout << "unknown";
  }
  std::cout << std::endl;
}

// Note: innermost dimension is at nDims - 1 (when nDims > 0)
void printStrides(const c10::VaryingStrides& strides) {
  std::cout << "Strides=(";
  for (size_t i = 0; i < *(strides.size()); ++i) {
    std::cout << *(strides[i]);
    if(i != *(strides.size())-1)
      std::cout << ", ";
    else
      std::cout << ")";
  }
}

void printSizes(const c10::VaryingShape& sizes) {
  std::cout << "Sizes=(";
  for (size_t i = 0; i < *(sizes.size()); ++i) {
    std::cout << *(sizes[i]);
    if(i != *(sizes.size())-1)
      std::cout << ", ";
    else
      std::cout << ")";
  }
}

void printCompleteTensor(const std::shared_ptr<c10::TensorType>& tensor) {
  std::cout << "Complete Tensor: ";
  std::cout << *(tensor->device()) << " ";
  std::cout << *(tensor->scalarType()) << " ";
  std::cout << "nDims: " << *(tensor->dim()) << " ";
  std::cout << std::endl;
  printSizes(tensor->sizes());
  std::cout << ", ";
  printStrides(tensor->strides());
  std::cout << std::endl;
  std::cout << "NonCollapsibleDims: " << ::torch::jit::fuser::getNumNonCollapsibleDims(tensor);
  std::cout << std::endl;
}

void printValue(const Value* const value) {
  if (value->isCompleteTensor()) {
    printCompleteTensor(value->type()->expect<TensorType>());
  } else if (value->type()->isSubtypeOf(NumberType::get())) {
    printScalar(value);
  } else {
    std::cout << "Request to print unknown value" << std::endl;
  }
}

// Returns true if and only if value is a scalar or a complete tensor type
bool validateValue(const Value* const value, const bool dbg = false) {
  if (dbg) {
    printValue(value);
  }

  if (value->isCompleteTensor() || value->type()->isSubtypeOf(NumberType::get())) {
    return true;
  }

  return false;
}

// Returns true if all inputs and outputs are complete tensors or scalars
// Note: complete tensor means device, nDims, sizes, and strides are known
// In particular, all optional values of sizes and strides have values
bool hasCompleteInputsAndOutputs(const Node* const node, const bool dbg = false) {
  const auto inputs = node->inputs();
  const auto outputs = node->outputs();

  if (dbg) {
    std::cout << "nInputs: " << inputs.size() << std::endl;
    std::cout << "nOutputs: " << outputs.size() << std::endl;
  }

  if (dbg) {
    std::cout << "Inputs: " << std::endl;
  }

  for (auto i = decltype(inputs.size()){0}; i < inputs.size(); ++i) {
    if (!validateValue(inputs[i], dbg)) {
      return false;
    }
  }

  if (dbg) {
    std::cout << "Outputs: " << std::endl;
  }

  for (auto i = decltype(outputs.size()){0}; i < outputs.size(); ++i) {
    if (!validateValue(outputs[i], dbg)) {
      return false;
    }
  }

  return true;
}

} // namespace


// Given a node and a fusion group, returns true if the node can be
// merged into the fusion group and false if it cannot.
// The fusion_group may be empty, specified by a nullptr.
bool mergeNodeWithFusionGroup(const Node* const node, int* fusion_key) {
  #if FUSER_DEBUG
    std::cout << "interface.cpp: mergeNodeWithFusionGroup()" << std::endl;
    const auto is_complete = hasCompleteInputsAndOutputs(node, true);
    std::cout << "is_complete: " << is_complete << std::endl;
  #else
    const auto is_complete = hasCompleteInputsAndOutputs(node, false);
  #endif // FUSER_DEBUG

  if (!is_complete) {
    return false;
  }

  // Validates that fusions:
  // (1) Have a single tensor output
  // (2) That all tensors are on the same device

  // Validates output
  const auto& outputs = node->outputs();

  if (outputs.size() != 1) {
    return false;
  }

  const auto& output = node->output();

  if (!(output->isCompleteTensor())) {
    return false;
  }

  // Acquires fusion device
  const std::shared_ptr<c10::TensorType> out_tensor = output->type()->expect<TensorType>();
  const auto fusion_device = *(out_tensor->device());

  #if FUSER_DEBUG
  std::cout << "Fusion Device: " << fusion_device <<std::endl;
  #endif // FUSER_DEBUG

  // Validate input tensors are all on fusion device
  const auto inputs = node->inputs();
  for (auto i = decltype(inputs.size()){0}; i < inputs.size(); ++i) {
    const std::shared_ptr<c10::TensorType> in_tensor = output->type()->expect<TensorType>();
    const auto input_device = *(in_tensor->device());
    if (fusion_device != input_device) {
      #if FUSER_DEBUG
        std::cout << "Input device != fusion device: " << input_device << std::endl;
      #endif // FUSER_DEBUG
      return false;
    }
  }

  if (fusion_device.type() == c10::kCPU) {
    #if FUSER_DEBUG
      std::cout << "Fusing on CPU" << std::endl;
    #endif // FUSER_DEBUG
    return torch::jit::fuser::cpu::mergeNodeWithFusionGroup(node, fusion_key);
  } else if (fusion_device.type() == c10::kCUDA) {
    #if FUSER_DEBUG
      std::cout << "Fusing on CUDA" << std::endl;
    #endif // FUSER_DEBUG
    return torch::jit::fuser::cuda::mergeNodeWithFusionGroup(node, fusion_key);
  } else {
    std::cout << "unknown fusion device: " << fusion_device << std::endl;
  }

  return false;
}

int fusion_counter = 0;
std::unordered_map<int, c10::DeviceType> fusion_to_device_map;

std::unordered_map<int, c10::DeviceType> getFusionToDeviceMap() {
  return fusion_to_device_map;
}

int getAndIncrementGlobalFusionCounter() {
  return fusion_counter++;
}

void callFusion(const int key, Stack& stack) {
  const auto device_type = fusion_to_device_map[key];
  std::cout << "Calling fusion on device " << device_type << std::endl;

  if (device_type == c10::kCPU) {
    torch::jit::fuser::cpu::callFusion(key, stack);
  }
}


// OLD STUFF BELOW HERE























int64_t registerFusion(const Node* fusion_group) {
  return fuser::registerFusion(fusion_group);
}

void runFusion(const int64_t key, Stack& stack) {
  const auto result = fuser::runFusion(key, stack);
  if (!result)
    fuser::runFallback(key, stack);
}

bool canFuseOnCPU() {
  return fuser::hasFusionBackend(at::DeviceType::CPU) &&
      detail::cpu_fuser_enabled;
}

bool canFuseOnGPU() {
  return fuser::hasFusionBackend(at::DeviceType::CUDA);
}

void overrideCanFuseOnCPU(bool value) {
  detail::cpu_fuser_enabled = value;
}

// Uses the above interface by stuffing the graph into a node and treating that
// node as a fusion group.
std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs) {
  // Creates a fusion group node
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group = wrapper_graph->insertNode(
      wrapper_graph->createWithSubgraph(prim::FusionGroup));
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // Creates the stack, registers and runs the fusion
  Stack stack = fmap<IValue>(inputs);
  const auto key = fuser::registerFusion(fusion_group);
  fuser::runFusion(key, stack);
  return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}

std::string debugGetFusedKernelCode(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs) {
  // Creates a fusion group node
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group =
      wrapper_graph->insertNode(wrapper_graph->createWithSubgraph(prim::FusionGroup));
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // Creates the stack, registers and runs the fusion
  Stack stack = fmap<IValue>(inputs);
  const auto key = fuser::registerFusion(fusion_group);

  std::string code;
  if (!fuser::runFusion(key, stack, &code)) {
    throw std::runtime_error("Could not run fusion for graph");
  }

  return code;
}

size_t nCompiledKernels() {
  return fuser::nCompiledKernels();
}

} // namespace jit
} // namespace torch
