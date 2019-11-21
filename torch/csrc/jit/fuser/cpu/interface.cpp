#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/cpu/interface.h>
#include <torch/csrc/jit/fuser/common/utils.h>

#include <asmjit/asmjit.h>
#include <iostream>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

using namespace torch::jit::fuser;

// Generated function (add) signature
typedef void (*addFunc)(unsigned, float*, float*, float*);

using namespace asmjit;
using namespace asmjit::x86;

namespace {
  std::unordered_map<int, addFunc> fusion_map;
}



// Returns true if the node is added to the fusion group, false o.w.
int tryCreateFusion(const Node* const node) {
  #if FUSER_DEBUG
    std::cout << "cpuMergeNodeWithFusionGroup" << std::endl;
  #endif // FUSER_DEBUG

  if (node->kind() != aten::add) {
    return -1;
  }

  // Creates a new fusion group

  // Validates inputs are fusible
  const auto inputs = node->inputs();
  const auto output = node->output();

  const auto lhs = inputs[0]->type()->expect<TensorType>();
  const auto rhs = inputs[1]->type()->expect<TensorType>();
  const auto c = inputs[2]; // TODO: validate c = 1

  const auto lhs_rank = getRank(lhs);
  const auto rhs_rank = getRank(rhs);

  if (lhs_rank != rhs_rank) {
    std::cout << "Rank mismatch!" << std::endl;
    return -1;
  }

  const auto lhs_logical_rank = getNumNonCollapsibleDims(lhs);
  const auto rhs_logical_rank = getNumNonCollapsibleDims(rhs);

  if (lhs_logical_rank != rhs_logical_rank) {
    std::cout << "Dims mismatch!" << std::endl;
    return -1;
  }

  if (lhs_logical_rank != 1) {
    std::cout << "Tensor's logical rank != 1!" << std::endl;
    return -1;
  }

  const auto lhs_innermost_stride = *(lhs->strides()[lhs_rank - 1]);
  const auto rhs_innermost_stride = *(rhs->strides()[rhs_rank - 1]);

  if (lhs_innermost_stride != 1 || rhs_innermost_stride != 1) {
    std::cout << "Innermost stride is not 1!" << std::endl;
    return -1;
  }

  const auto lhs_numel = getNumel(lhs);
  const auto rhs_numel = getNumel(rhs);

  if (lhs_numel != rhs_numel) {
    std::cout << "numel mismatch!" << std::endl;
    return -1;
  }

  // Creates fusion_group

  // Creates runtime and assembler
  JitRuntime rt;
  CodeHolder code;
  code.init(rt.codeInfo());
  Assembler a(&code);

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

  const auto key = ::torch::jit::getAndIncrementGlobalFusionCounter();

  // Updates maps
  fusion_map[key] = fn;
  ::torch::jit::getFusionToDeviceMap()[key] = c10::kCPU;

  return key;
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
