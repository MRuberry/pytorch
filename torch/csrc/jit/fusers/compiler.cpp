#include "torch/csrc/jit/fusers/compiler.h"

#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/type.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/cache.h"
#include "torch/csrc/jit/fusers/simple_mappable.h"

#if USE_CUDA_FUSER
  #include "torch/csrc/jit/fusers/cuda/codegen.h"
  #include "torch/csrc/jit/fusers/cuda/compiler.h"
#endif // USE_CUDA_FUSER

#if USE_CPU_FUSER
  #include "torch/csrc/jit/fusers/cpu/codegen.h"
#endif // USE_CPU_FUSER

#include <iostream>
#include <memory>
#include <unordered_set>
#include <utility>
#include <string>
#include <atomic>
#include <sstream>
#include <stdexcept>
#include <iostream> // TODO: remove me

namespace torch { namespace jit { namespace fusers {

static std::atomic<size_t> next_kernel_id{0};

// // If the input is used by a single chunk node, returns that node
// // Returns nullptr otherwise
// static Node* usedInFusedChunk(const Value* const input) {
//   auto uses = input->uses();
//   if (uses.size() == 1) {
//     Node *user = uses[0].user;
//     if (user->kind() == prim::ConstantChunk) {
//       return user;
//     }
//   }
//   return nullptr;
// }

// static std::vector<PartitionInfo> getInputChunkDescriptors(std::shared_ptr<Graph> graph) {
//   std::vector<PartitionInfo> descs;
//   descs.reserve(graph->inputs().size());
//   for (Value* input : graph->inputs()) {
//     if (Node* chunk = usedInFusedChunk(input)) {
//       descs.emplace_back(chunk->i(attr::chunks), chunk->i(attr::dim));
//     } else {
//       descs.emplace_back(1, 0);
//     }
//   }
//   return descs;
// }

// static std::vector<int64_t> getInputDependencies(Value* output) {
//   // Run a DFS traversal to find all inputs that affect a given output value
//   std::vector<Value*> queue{output};
//   std::unordered_set<Value*> inputs;
//   std::unordered_set<Value*> seen;
//   while (!queue.empty()) {
//     Value* val = queue.back(); queue.pop_back();
//     Node* producer = val->node();
//     if (producer->kind() == prim::Param) {
//       inputs.insert(val);
//       continue;
//     }
//     for (Value* input : producer->inputs()) {
//       if (/*bool inserted = */seen.insert(input).second) {
//         queue.push_back(input);
//       }
//     }
//   }

//   // Convert Value* into offsets into the graph's input list
//   std::vector<int64_t> offsets;
//   offsets.reserve(inputs.size());
//   for (Value* input : inputs) {
//     offsets.push_back(input->offset());
//   }

//   std::sort(offsets.begin(), offsets.end());
//   return offsets;
// }

// static std::vector<std::vector<int64_t>> getInputBroadcastGroups(std::shared_ptr<Graph> graph) {
//   std::unordered_set<std::vector<int64_t>, torch::hash<std::vector<int64_t>>> broadcast_groups;
//   for (Value* output : graph->outputs()) {
//     broadcast_groups.insert(getInputDependencies(output));
//   }
//   return std::vector<std::vector<int64_t>>{broadcast_groups.begin(), broadcast_groups.end()};
// }

// // The place for device independent "upfront" or "storage-based" compilation
// // Note: type and size information is not available at this time. 
// static bool upfrontCompilation(FusionSpec& spec) {
//   const auto broadcast_groups = getInputBroadcastGroups(spec.graph());
//   const auto chunk_descs = getInputChunkDescriptors(spec.graph());
//   spec.setInputBroadcastGroups(broadcast_groups);
//   spec.setinputChunkDescriptors(chunk_descs);
//   return true;
// }

// // curDimIndex = linearId % sizes[i]; // % sizes[i] is not needed for d == 0, because we already guard for numel outside the index calculation
// // offset += curDimIndex*strides[i]; // *strides[i] is optional if list_is_cont becaause strides.back() == 1
// // linearId /= sizes[i];
// //printf("tensor ${tensor} sizes[${d}] = %d, strides[${d}] = %d\n", ${tensor}.sizes[${d}],${tensor}.strides[${d}]);
// const auto dim_calc_template = CodeTemplate(R"(
// size_t ${tensor}_dimIndex${d} = ${tensor}_linearIndex ${mod_sizes};
// ${tensor}_offset += ${tensor}_dimIndex${d} ${times_stride};
// )");

// // Returns a C-string representation of the given type. 
// // Note: "Half" is specialized to prevent "at::Half" from appearing.
// static const char* scalarTypeName(at::ScalarType type) {
//   if (type == at::ScalarType::Half) {
//     return "half";
//   }

//   switch(type) {
//     #define DEFINE_CASE(ctype,name,_) \
         case at::ScalarType::name: return #ctype;
//     AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(DEFINE_CASE)
//     #undef DEFINE_CASE
//     default:
//       throw std::runtime_error("unknown scalar type");
//   }
// }

// static void emitIndexingFor(
//   std::ostream& out
// , const std::string& tensor
// , int ndim
// , bool last_is_cont) {
//   std::cout << "emitIndexingFor()" << std::endl;
//   TemplateEnv env;
//   env.s("tensor",tensor);
//   out << format("IndexType ${tensor}_offset = 0;\n",env);
//   out << format("IndexType ${tensor}_linearIndex = linearIndex;\n",env);
//   for (int d = ndim - 1; d >= 0; --d) {
//     env.d("d",d);
//     env.s("mod_sizes", d > 0 ? format("% ${tensor}.sizes[${d}]",env) : "");
//     env.s("times_stride",(d < ndim - 1 || !last_is_cont) ?
//       format("* ${tensor}.strides[${d}]",env) : "");
//     out << dim_calc_template.format(env);
//     if (d > 0) {
//       out << format("${tensor}_linearIndex /= ${tensor}.sizes[${d}];\n",env);
//     }
//   }
//   std::cout << "emitIndexingFor() end" << std::endl;
// }

// // TODO: handle cases where we need to generate > 2^32 element tensors
// bool compileFusion(
//   const FusionSpec& spec
// , const FusionArgSpec& arg_spec
// , const std::vector<int64_t>& map_size) {
//   // TODO: debugging, remove me
//   std::cout << "compileFusion()" << std::endl;
//   const auto& graph = *spec.graph();
//   const auto& input_desc = arg_spec.descs();

//   // Creates CompleteTensorType outputs for graph outputs
//   // Also infers device from first output
//   // Note: adjusts output sizes for special-case of concat
//   std::vector<TensorDesc> output_desc;
//   auto device = kCPUDevice;
//   for (Value* output : spec.graph()->outputs()) {
//     auto sizes = map_size;
//     if (output->node()->kind() == prim::FusedConcat) {
//       sizes.at(output->node()->i(attr::dim)) *= output->node()->inputs().size();
//     }
//     std::cout << output->type()->str() << std::endl;
//     if (auto tt = output->type()->cast<TensorType>()) {
//       device = tt->device();
//       auto ctt = CompleteTensorType::create(tt->scalarType(), tt->device(), sizes);
//       output_desc.emplace_back(std::move(ctt));
//     } else {
//       // TODO: assertion (should be checked already)
//     }
//   }

//     // TODO: remove me
//   std::cout << "compileFusion() A" << std::endl;

//   const bool useCUDA = (device != kCPUDevice);

//   // Short-circuits if fusion is disabled for the requested device
//   if (useCUDA && !canFuseOnGPU()) return false;
//   if (!useCUDA && !canFuseOnCPU()) return false;

//   // Generates kernel environment
//   TemplateEnv env;
//   std::stringstream body;
//   std::stringstream tensorOffsets;
//   std::vector<std::string> formals;
//   std::vector<std::string> argument_loads;

//   const auto kernel_name = "kernel_" + std::to_string(next_kernel_id++);
//   env.s("kernelName", kernel_name);
//   env.s("IndexType","unsigned int"); //avoiding slow header includes to get uint32_t
  
//   // Defines emitFormal() lambda
//   auto emitFormal = [&](const Value* const n, const TensorDesc& desc) {
//     std::cout << "emitFormal()" << std::endl;
//     std::string tensor = "t" + std::to_string(formals.size()); //can't be unique() because Param may be an output
//     std::cout << "emitFormal() before nDim acquisition" << std::endl;
//     const auto nDim = desc.nDim();
//     std::cout << "emitFormal() before emitIndexingFor" << std::endl;
//     emitIndexingFor(tensorOffsets, tensor, nDim,  desc.lastIsContiguous());
//     env.s("tensor", tensor);
//     env.d("formal_index", formals.size() + 1); // + 1 because the first argument is the linearIndex
//     std::cout << "emitFormal() middle" << std::endl;
//     env.d("nDim", nDim);
//     env.s("scalar_type", scalarTypeName(desc.scalar_type));
//     formals.push_back(format("TensorInfo<${scalar_type},${nDim}> ${tensor}", env));
//     argument_loads.push_back(format("*static_cast<TensorInfo<${scalar_type},${nDim}>*>(args[${formal_index}])", env));
//     std::cout << "emitFormal() end" << std::endl;
//   };

//   // Writes input args
//   std::vector<std::pair<const Value*, const TensorDesc&>> flat_inputs;
//   std::vector<PartitionDesc> chunk_desc;
//   { 
//     size_t input_index = 0;
//     for(auto p : graph.inputs()) {
//       if (Node* chunk = usedInFusedChunk(p)) {
//         int64_t dim = chunk->i(attr::dim);
//         int64_t chunks = chunk->i(attr::chunks);
//         chunk_desc.emplace_back(input_desc[input_index++], chunks, dim);
//         for (auto * o : chunk->outputs()) {
//           flat_inputs.emplace_back(o, *chunk_desc.back().subTensorDesc);
//         }
//       } else {
//         chunk_desc.emplace_back();
//         flat_inputs.emplace_back(p, input_desc[input_index++]);
//       }
//     }
//     for (auto& input : flat_inputs) {
//       emitFormal(input.first, input.second);
//     }
//   }

//       // TODO: remove me
//   std::cout << "compileFusion() B" << std::endl;

//   // Writes output args
//   std::vector<std::pair<const Value*, const TensorDesc>> flat_output_nodes;
//   std::vector<PartitionDesc> concat_desc;
//   {
//     size_t i = 0;
//     for (auto o : graph.outputs()) {
//       auto& desc = output_desc[i++];
//       std::cout << "before node() acquisition" << std::endl;
//       if (o->node()->kind() != prim::FusedConcat) {
//         std::cout << "after node() acquisition" << std::endl;
//         emitFormal(o, desc);
//         concat_desc.emplace_back();
//         flat_output_nodes.emplace_back(o, desc);
//         std::cout << "end of if block" << std::endl;
//       } else {
//         std::cout << "in else block" << std::endl;
//         auto cat = o->node();
//         concat_desc.emplace_back(desc, cat->inputs().size(), cat->i(attr::dim));
//         for(auto c : cat->inputs()) {
//           emitFormal(c, *concat_desc.back().subTensorDesc);
//           flat_output_nodes.emplace_back(c, desc);
//         }
//       }
//     }
//   }

//         // TODO: remove me
//   std::cout << "compileFusion() BA" << std::endl;

//   // Writes input access code (in body)
//   // Note: records if a half tensor is seen so that the half header can be 
//   // included later.
//   // Note: half tensors only supported in CUDA fusions.
//   size_t formal_count = 0;
//   bool has_half_tensor = false;
//   for(auto input : flat_inputs) {
//     auto p = input.first;
//     env.s("node", valueName(p));
//     env.d("formal", formal_count++);

//     // Acquires and converts (if needed) inputs
//     bool is_half = (input.second.scalar_type == at::ScalarType::Half);
//     if (is_half) {
//       env.s(
//         "access"
//       , format("__half2float(t${formal}.data[t${formal}_offset])", env));
//       has_half_tensor = true;
//     } else {
//       env.s("access", format("t${formal}.data[t${formal}_offset]", env));
//     }

//     //TODO: actual type propagation rather than relying on auto..
//     body << format("auto ${node} = ${access};\n", env);
//   }

//         // TODO: remove me
//   std::cout << "compileFusion() BC" << std::endl;

//   // Writes operations (in body)
//   // Note: random ops only supported in CUDA fusions
//   bool has_random = false;
//   for (const auto& n : graph.nodes()) {
//     if (n->kind() == prim::FusedConcat) continue;
//     if (n->kind() == prim::ConstantChunk) continue;
//     if (n->kind() == aten::rand_like) has_random = true;
//     env.s("node", valueName(n->output()));
//     env.s("rhs", encodeRHS(n));
//     body << format("auto ${node} = ${rhs};\n", env);
//   }

//       // TODO: remove me
//   std::cout << "compileFusion() C" << std::endl;

//   // Writes output access code
//   for (auto output : flat_output_nodes) {
//     auto o = output.first;
//     env.d("formal", formal_count++);
//     env.s("access", format("t${formal}.data[t${formal}_offset]", env));
//     env.s("node", valueName(o));

//     // Acquires and converts (if needed) outputs
//     bool is_half = output.second.scalar_type == at::ScalarType::Half;
//     if (is_half && !useCUDA) return false; 
//     if (is_half) {
//       body << format("${access} = __float2half(${node});\n", env);
//       has_half_tensor = true;
//     } else {
//       body << format("${access} = ${node};\n", env);
//     }
//   }

//   // Includes headers
//   env.s("tensorOffsets", tensorOffsets.str());
//   env.s("kernelBody", body.str());
//   env.v("formals", formals);
//   env.v("argument_loads", argument_loads);

//   // TODO: remove me
//   std::cout << "device-specific codegen" << std::endl;

//   // Finishes code generation with device-specific codegen
//   std::stringstream ss;
//   bool codegen_result = false;
//   bool compile_result = false;
//   if (useCUDA) { 
//     #if USE_CUDA_FUSER
//       codegen_result = cuda::generateCUDA(ss, env, has_half_tensor, has_random);
//       const auto code = ss.str();
//       compile_result = cuda::compileCUDA(device, kernel_name, code);
//     #else // !USE_CUDA_FUSER
//       throw std::runtime_error("CUDA declared fusable but PyTorch was not built with CUDA fuser.");
//     #endif // USE_CUDA_FUSER
//   } else { // CPU compilation
//     #if USE_CPU_FUSER
//       codegen_result = cpu::generateCpp(ss, env, has_half_tensor, has_random);
//     #else // !USE_CPU_FUSER
//       throw std::runtime_error("Cpp declared fusable but PyTorch was not built with CPU fuser.");
//     #endif // USE_CPU_FUSER
//   }


// //   return std::make_tuple(std::move(chunk_desc), std::move(concat_desc), has_random);
// // }

//   return false;
// }

bool isSupportedOp(const Node* node) {
  return (node->kind() == prim::Constant || isSimpleMap(node));
}

ReturnCode registerFusion(int64_t& key, const Node* fusion_group) {
  // Creates FusionSpec
  auto graph = fusion_group->g(attr::Subgraph)->copy();

  // Creates and stores the fusion spec
  EraseShapeInformation(*graph);
  key = store(graph);
  // const auto maybe_spec = retrieve(key);

  // if (!maybe_spec) {
  //   // TODO: error out
  // }

  // Validates the graph
  bool is_fusable = true;
  for (const auto& node : graph->nodes()) {
    if (!::torch::jit::fusers::isSupportedOp(node)) {
      std::cout << node->kind().toDisplayString() << std::endl;
      is_fusable = false;
      break;
    }
  }

  if (!is_fusable) {
    const auto maybe_spec = retrieve(key);
    if (!maybe_spec) 
      throw std::runtime_error("Registered fusion specification not found.");
    maybe_spec->setFusable(false);
    return ReturnCode::UNSUPPORTED_OP;
  }
  
  // // Performs device-independent upfront compilation of the spec 
  // // if (canFuseOnCPU() || canFuseOnGPU())
  // //   upfrontCompilation(*maybe_spec);

  return ReturnCode::SUCCESS;
}

} // namespace fusers
} // namespace jit
} // namespace torch
