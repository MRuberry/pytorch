#include "torch/csrc/jit/fusers/fallback.h"

#include "torch/csrc/jit/interpreter.h"

namespace torch { namespace jit { namespace fusers {

void runFallback(Node* fusion_group, Stack& stack) {
  auto graph = fusion_group->g(attr::Subgraph)->copy();
  Code code{graph};
  InterpreterState{code}.run(stack);
}

} // namespace fusers
} // namespace jit
} // namespace torch
