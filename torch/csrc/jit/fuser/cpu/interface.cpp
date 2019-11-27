#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/cpu/interface.h>
#include <torch/csrc/jit/fuser/common/utils.h>

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

// Generated function (add) signature
// TODO: don't compile-time generate signatures
typedef void (*addFunc)(unsigned, float*, float*, float*);

using namespace asmjit;
using namespace asmjit::x86;

namespace {
  std::unordered_map<int, addFunc> fusion_map;
  std::unordered_set<Symbol> fusibleNodes{
    aten::add
  };

  // Returns true iff the node is fusible
  // TODO: The set of fusible nodes is fixed at compile time, but
  //  we're using a mutable data structure
  bool isFusible(const Node* const node) {
    // Checks that node kind is in fusibleNodes
    const auto kind = node->kind();
    auto it = fusibleNodes.find(kind);
    if (it == fusibleNodes.end()) {
      return false;
    }

    // Check the following:
    //  - the node's output is a single contiguous tensor
    //  - the inputs are either tensors with the same shape as the output or
    //      scalars == 1.f
    //  - the first two inputs are tensors and the third is a scalar
    // TODO: relax these requirements (requires updating fusion logic)
    const auto output = node->output()->type()->expect<TensorType>();
    const auto inputs = node->inputs();

    for (auto i = decltype(inputs.size()){0}; i < inputs.size(); ++i) {
      const auto* input = inputs[i];
      // If a (complete) tensor, checks that it's collapsible to 1 dim
      if (input->isCompleteTensor()) {
        const auto tensor_input = input->type()->expect<TensorType>();
        TensorMeta meta = torch::jit::fuser::collapse(output, tensor_input);
        torch::jit::fuser::printMeta(meta);
        if (meta.rank() != 1) {
          return false;
        }
      } else if (input->type()->isSubtypeOf(NumberType::get())) {
        const auto scalar = getAsFloat(input);
        if (scalar != 1.f) {
          return false;
        }
      } else {
        TORCH_CHECK(false, "Input has unknown type");
      }
    }

    // Checks that first two inputs are tensors and third is scalar
    const auto* lhs = inputs[0];
    const auto* rhs = inputs[1];
    const auto* c = inputs[3];
    if (!lhs->isCompleteTensor()
     || !rhs->isCompleteTensor()
     || !c->type()->isSubtypeOf(NumberType::get())) {
      return false;
    }

    return true;
  }
}

// Returns the key of the new fusion or -1 if the node is not fusible
// TODO: tracking via keys requires thread-safety
// TODO: associate fusion metadata with key during construction to support
//  fusion kinds
int tryCreateFusion(const Node* const node) {
  #if FUSER_DEBUG
    std::cout << "cpuMergeNodeWithFusionGroup" << std::endl;
  #endif // FUSER_DEBUG

  if (!isFusible(node)) {
    std::cout << "node is not fusible!" << std::endl;
    return -1;
  }
  std::cout << "node is fusible!" << std::endl;

  // Returns the key corresponding to this fusion
  return ::torch::jit::getAndIncrementGlobalFusionCounter();
}

void compileFusion(const Node* const fusion) {

}



void callFusion(const int key, Stack& stack) {
  // auto& fn = fusion_map[key];
  auto inputs = last(stack, 3);
  auto lhs = inputs[0].toTensor();
  auto rhs = inputs[1].toTensor();

  std::cout << "lhs numel: " << lhs.numel() << std::endl;

  auto output = at::empty_like(lhs, lhs.options(), c10::MemoryFormat::Preserve);

  std::cout << "output numel: " << output.numel() << std::endl;

  float* out_storage = output.storage().data<float>();
  float* lhs_storage = lhs.storage().data<float>();
  float* rhs_storage = rhs.storage().data<float>();

  std::cout << "out_storage: " << out_storage << std::endl;
  std::cout << "lhs_storage: " << lhs_storage << std::endl;
  std::cout << "rhs_storage: " << rhs_storage << std::endl;
  unsigned nElements = lhs.numel();

  // TODO: remove me
  // Need to manage lifetime of these objects better
  // Creates runtime and assembler
  JitRuntime rt;
  CodeHolder code;
  code.init(rt.codeInfo());
  Assembler a(&code);


  // CodeHolder proto_code;
  // Compiler cc(&proto_code);
  // const auto argCount = 3; // out, lhs, rhs
  // FuncSignatureBuilder signature(CallConv::kIDHost);
  // signature.setRetT<void>();
  // for (auto i = decltype(argCount){0}, i < argCount; ++i) {
  //   signature.addArgT<float*>();
  // }
  // auot* proto_fn = cc.addFunc(signature);
  // cc.finalize();




    // Creates fusion (element-by-element contiguous add)
    Label LoopInc = a.newLabel();
    Label LoopBody = a.newLabel();
    Label Exit = a.newLabel();

    // Short-circuits on size == 0
    a.test(edi, edi);
    a.je(Exit);

    // Stores # of loop iterations, sets loop counter to zero
    a.lea(r8d, dword_ptr(rdi, - 1)); // r8 = size - 1
    a.xor_(eax, eax); // clears eax
    a.jmp(LoopBody); // do () { } while () loop form

    // Loop incrementer
    a.bind(LoopInc);
    a.mov(rax, rdi); // offset = offset + 1

    // Loop body
    a.bind(LoopBody);
    a.vmovss(xmm0, dword_ptr(rdx, rax, 2)); // xmm0 = lhs[offset]
    a.vaddss(xmm0, xmm0, dword_ptr(rcx, rax, 2)); // xmm0 = xmm0 + rhs[offset]
    a.lea(rdi, dword_ptr(rax, 1)); // size = offset + 1
    a.vmovss(dword_ptr(rsi, rax, 2), xmm0); // out[offset] = xmm0

    // Checks if loop is finished
    a.cmp(rax, r8); // if offset == size - 1, terminate
    a.jne(LoopInc);

    // Exit
    a.bind(Exit);
    a.ret();

    // Jits the code and stores in function
    addFunc fn;
    Error err = rt.add(&fn, &code);
    if (err) {
      std::cout << "Error while jitting!" << std::endl;
    }

  fn(nElements, out_storage, lhs_storage, rhs_storage);

  // Updates stack
  drop(stack, 3);
  push_one(stack, output);
}

}}}} // namespace torch::jit::fuser::cpu
