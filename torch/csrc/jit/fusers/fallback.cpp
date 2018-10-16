#include "torch/csrc/jit/fusers/fallback.h"

#include "torch/csrc/utils/functional.h" //fmap
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/fusers/cache.h"

#include <iostream> // TODO: remove me

namespace torch { namespace jit { namespace fusers {

// Registers fused operators so that fused graphs can properly generate fallback code.
RegisterOperators reg_fused_operators({
  Operator(
    prim::FusedConcat,
    [](Node* node) {
      int64_t dim = node->i(attr::dim);
      int64_t num_inputs = node->inputs().size();
      return [dim, num_inputs](Stack& stack) {
        auto result = at::cat(
          fmap(last(stack, num_inputs), [](const IValue& i) { return i.toTensor(); }),
          dim
        );
        drop(stack, num_inputs);
        pack(stack, std::move(result));
        return 0;
      };
    })
});

void runFallback(int64_t key, Stack& stack) {
  // TODO: debugging only remove me
  std::cout << "runFallback()" << std::endl;
  auto maybe_spec = retrieve(key);

  if (!maybe_spec) {
    // TODO: error out
  }

  InterpreterState{(*maybe_spec).code()}.run(stack);
}

} // namespace fusers
} // namespace jit
} // namespace torch
