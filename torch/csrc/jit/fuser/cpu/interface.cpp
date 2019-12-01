#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/cpu/interface.h>
#include <torch/csrc/jit/fuser/cpu/ir.h>
#include <torch/csrc/jit/fuser/common/utils.h>
#include <torch/csrc/jit/fuser/common/tensor_meta.h>

#include <c10/util/Exception.h>

#include <asmjit/asmjit.h>

#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

using namespace torch::jit::fuser;

std::unordered_set<Symbol> fusibleNodes{
  aten::add
};

bool isFusibleOnCPU(const Node* const node) {
  const auto it = fusibleNodes.find(node->kind());
  if (it == fusibleNodes.end()) {
    return false;
  }

  return true;
}


  // Returns true iff the node is fusible
  // TODO: The set of fusible nodes is fixed at compile time, but
  //  we're using a mutable data structure
  // TODO: expand set of what's fusible
//   bool isFusible(const Node* const node) {
//     // Checks that node kind is in fusibleNodes
//     const auto kind = node->kind();
//     const auto it = fusibleNodes.find(kind);
//     if (it == fusibleNodes.end()) {
//       return false;
//     }

//     // Note: assumes single output fusion
//     const auto output = node->output()->type()->expect<TensorType>();
//     const auto inputs = node->inputs();

//     const auto loop_metas = torch::jit::fuser::getLoopMetas(node->outputs(), inputs);

//     // DEBUG SPEW
//     torch::jit::fuser::printMeta(std::cout, loop_metas[0]);

//     // Note: only rank 1 fusions
//     // TODO: expand fusible set
//     if (loop_metas[0].rank() != 1) {
//       return false;
//     }

//     // Checks that first two inputs are tensors and third is scalar
//     // TODO: expand fusible set
//     const auto* lhs = inputs[0];
//     const auto* rhs = inputs[1];
//     const auto* c = inputs[2];
//     if (!lhs->isCompleteTensor()
//      || !rhs->isCompleteTensor()
//      || !c->type()->isSubtypeOf(NumberType::get())) {
//       return false;
//     }

//     return true;
//   }
// }

// // Returns the key of the new fusion or -1 if the node is not fusible
// // TODO: tracking via keys requires thread-safety
// // TODO: associate fusion metadata with key during construction to support
// //  fusion kinds
// int tryCreateFusion(const Node* const node) {
//   #if FUSER_DEBUG
//     std::cout << "cpuMergeNodeWithFusionGroup" << std::endl;
//   #endif // FUSER_DEBUG

//   if (!isFusible(node)) {
//     std::cout << "node is not fusible!" << std::endl;
//     return -1;
//   }
//   std::cout << "node is fusible!" << std::endl;

//   // Returns the key corresponding to this fusion
//   return ::torch::jit::getAndIncrementGlobalFusionCounter();
// }

// //TODO: assumes all fusions are loop fusions
// void compileFusion(const Node* const fusion) {
//   std::cout << "compileFusion()" << std::endl;
//   TORCH_CHECK(fusion->kind() == prim::FusionGroup, "Trying to compile a node that's not a fusion group!");

//   const auto key = fusion->i(attr::value);
//   Graph& graph = *fusion->g(attr::Subgraph);
//   const Graph& const_graph = *fusion->g(attr::Subgraph);

//   const auto inputs = graph.inputs();
//   const auto outputs = graph.outputs();

//   // Creates kernel IR
//   // Kernel* k = new Kernel;



//   // TODO: stuff scalar types
//   // Unpacks outputs, inputs
//   // std::unordered_map<Value*, Register*> value_to_register_map;
//   // std::vector<Value*> tensor_values;
//   // int unpack_counter = 0;
//   // for (auto* output : outputs) {
//   //   PointerRegister* output_register = new PointerRegister{k->getRegisterName()};
//   //   value_to_register_map[output] = output_register;
//   //   UnpackPointer* unpack_pointer = new UnpackPointer{output_register, unpack_counter++};
//   //   k->addNode(unpack_pointer);
//   //   tensor_values.push_back(output);
//   // }
//   // for (auto* input : inputs) {
//   //   if (torch::jit::fuser::isScalar(input)) {
//   //     ScalarRegister* input_register = new ScalarRegister{k->getRegisterName()};
//   //     value_to_register_map[input] = input_register;
//   //     UnpackScalar* unpack_scalar = new UnpackScalar{input_register, unpack_counter++};
//   //     k->addNode(unpack_scalar);
//   //   } else {
//   //     PointerRegister* input_register = new PointerRegister{k->getRegisterName()};
//   //     value_to_register_map[input] = input_register;
//   //     UnpackPointer* unpack_pointer = new UnpackPointer{input_register, unpack_counter++};
//   //     k->addNode(unpack_pointer);
//   //     tensor_values.push_back(input);
//   //   }
//   // }

//   // // Computes innermost dim
//   // const auto metas = getLoopMetas(const_graph.outputs(), const_graph.inputs());
//   // const auto out_meta = metas[0];
//   // TORCH_CHECK(out_meta.rank() == 1, "Trying to compile but out rank != 1");
//   // const auto inner_loop_size = out_meta.sizes()[0];

//   // // Emits innermost loop
//   // constexpr int innermost_stride = 1;
//   // Loop* inner_loop = new Loop{0, inner_loop_size, innermost_stride};
//   // k->addNode(inner_loop);

//   // // Emits operations
//   // auto it = graph.block()->nodes().begin();
//   // while (it != graph.block()->nodes().end()) {
//   //   auto* node = *it;

//   //   // TODO: lower with scalar (requires mul)
//   //   // TODO: lower with store!
//   //   if (node->kind() == aten::add) {
//   //     auto* output = node->output();
//   //     auto* lhs = node->inputs()[0];
//   //     auto* rhs = node->inputs()[1];
//   //     auto* c = node->inputs()[2];

//   //     // Moves tensor values into registers
//   //     auto* lhs_register = value_to_register_map[lhs];
//   //     auto* rhs_register = value_to_register_map[rhs];
//   //     auto* c_register = value_to_register_map[c];

//   //     auto lhs_kind = lhs_register->kind();
//   //     // Loads pointed to values into a register
//   //     // TODO: make these snippets modular functions
//   //     ScalarRegister* actual_lhs_register;
//   //     if (lhs_kind == RegisterKind::Pointer) {
//   //       actual_lhs_register = new ScalarRegister{k->getRegisterName()};
//   //       Move* move = new Move{actual_lhs_register, static_cast<PointerRegister*>(lhs_register)};
//   //       k->addNode(move);
//   //     } else if (lhs_kind == RegisterKind::Scalar) {
//   //       actual_lhs_register = static_cast<ScalarRegister*>(lhs_register);
//   //     } else {
//   //       TORCH_CHECK(false, "Unhandled (like constant) scalar type");
//   //     }

//   //     auto rhs_kind = lhs_register->kind();
//   //     ScalarRegister* actual_rhs_register;
//   //     if (rhs_kind == RegisterKind::Pointer) {
//   //       actual_rhs_register = new ScalarRegister{k->getRegisterName()};
//   //       Move* move = new Move{actual_rhs_register, static_cast<PointerRegister*>(rhs_register)};
//   //       k->addNode(move);
//   //     } else if (lhs_kind == RegisterKind::Scalar) {
//   //       actual_rhs_register = static_cast<ScalarRegister*>(rhs_register);
//   //     } else {
//   //       TORCH_CHECK(false, "Unhandled (like constant) scalar type");
//   //     }

//   //     ScalarRegister* output_register = new ScalarRegister{k->getRegisterName()};

//   //     Add* add = new Add{
//   //       output_register
//   //     , actual_lhs_register
//   //     , actual_rhs_register};
//   //     k->addNode(add);

//   //     // Updates Value->Register map OR stores output
//   //     if (value_to_register_map.find(output) == value_to_register_map.end()) {
//   //       value_to_register_map[output] = output_register;
//   //     } else {
//   //       auto* existing_output_register = value_to_register_map[output];
//   //       Move* move = new Move{existing_output_register, output_register};
//   //       k->addNode(move);
//   //     }
//   //   }

//   //   ++it;
//   // }

//   // // Emits tensor offset updates
//   // ConstantRegister* stride_register = new ConstantRegister{k->getRegisterName(), 1};
//   // for (auto* tensor_value : tensor_values) {

//   //   Add* add = new Add{
//   //     value_to_register_map[tensor_value]
//   //   , value_to_register_map[tensor_value]
//   //   , stride_register};

//   //   k->addNode(add);
//   // }

//   // // TODO: loop must be lowered to label + jmp
//   // // TODO: when tensors are broadcasting their innermost dim
//   // //  hoist those loads outside the loop and (not just innermost)

//   // k->print(std::cout);





//   // allocates registers for outputs and inputs
//   // const Register* const stack_vector = new Register{k->getName()};
//   // FromStack* from_stack = new FromStack{stack_vector, inputs.size()};
//   // AllocateOutputs* allocate_outputs = new AllocateOutputs;

//   // Iter* loop_start = new Iter{0, 1, 1};

//   // k->addNode(from_stack);
//   // k->addNode(allocate_outputs);
//   // k->addNode(loop_start);
// }

// void callFusion(const int key, Stack& stack) {
//   // auto& fn = fusion_map[key];
//   auto inputs = last(stack, 3);
//   auto lhs = inputs[0].toTensor();
//   auto rhs = inputs[1].toTensor();

//   std::cout << "lhs numel: " << lhs.numel() << std::endl;

//   auto output = at::empty_like(lhs, lhs.options(), c10::MemoryFormat::Preserve);

//   std::cout << "output numel: " << output.numel() << std::endl;

//   float* out_storage = output.storage().data<float>();
//   float* lhs_storage = lhs.storage().data<float>();
//   float* rhs_storage = rhs.storage().data<float>();

//   std::cout << "out_storage: " << out_storage << std::endl;
//   std::cout << "lhs_storage: " << lhs_storage << std::endl;
//   std::cout << "rhs_storage: " << rhs_storage << std::endl;
//   unsigned nElements = lhs.numel();

//   // TODO: remove me
//   // Need to manage lifetime of these objects better
//   // Creates runtime and assembler
//   JitRuntime rt;
//   CodeHolder code;
//   code.init(rt.codeInfo());
//   Assembler a(&code);


//   // CodeHolder proto_code;
//   // Compiler cc(&proto_code);
//   // const auto argCount = 3; // out, lhs, rhs
//   // FuncSignatureBuilder signature(CallConv::kIDHost);
//   // signature.setRetT<void>();
//   // for (auto i = decltype(argCount){0}, i < argCount; ++i) {
//   //   signature.addArgT<float*>();
//   // }
//   // auot* proto_fn = cc.addFunc(signature);
//   // cc.finalize();




//     // Creates fusion (element-by-element contiguous add)
//     Label LoopInc = a.newLabel();
//     Label LoopBody = a.newLabel();
//     Label Exit = a.newLabel();

//     // Short-circuits on size == 0
//     a.test(edi, edi);
//     a.je(Exit);

//     // Stores # of loop iterations, sets loop counter to zero
//     a.lea(r8d, dword_ptr(rdi, - 1)); // r8 = size - 1
//     a.xor_(eax, eax); // clears eax
//     a.jmp(LoopBody); // do () { } while () loop form

//     // Loop incrementer
//     a.bind(LoopInc);
//     a.mov(rax, rdi); // offset = offset + 1

//     // Loop body
//     a.bind(LoopBody);
//     a.vmovss(xmm0, dword_ptr(rdx, rax, 2)); // xmm0 = lhs[offset]
//     a.vaddss(xmm0, xmm0, dword_ptr(rcx, rax, 2)); // xmm0 = xmm0 + rhs[offset]
//     a.lea(rdi, dword_ptr(rax, 1)); // size = offset + 1
//     a.vmovss(dword_ptr(rsi, rax, 2), xmm0); // out[offset] = xmm0

//     // Checks if loop is finished
//     a.cmp(rax, r8); // if offset == size - 1, terminate
//     a.jne(LoopInc);

//     // Exit
//     a.bind(Exit);
//     a.ret();

//     // Jits the code and stores in function
//     addFunc fn;
//     Error err = rt.add(&fn, &code);
//     if (err) {
//       std::cout << "Error while jitting!" << std::endl;
//     }

//   fn(nElements, out_storage, lhs_storage, rhs_storage);

//   // Updates stack
//   drop(stack, 3);
//   push_one(stack, output);
// }

}}}} // namespace torch::jit::fuser::cpu
