#pragma once

#include <string>
#include <vector>

namespace torch { namespace jit { namespace fusers {

enum ReturnCode {
  SUCCESS = 0
, NOT_IMPL
, FUSION_DISABLED
, UNSUPPORTED_OP
, NONFUSABLE_SPEC
, NO_INPUTS
, NON_FLOATING_INPUT
, NO_OUTPUTS
, NO_MAP_SIZE
};

const static std::vector<std::string> code_strings = {
  "SUCCESS"
, "NOT_IMPL"
, "FUSION_DISABLED"
, "UNSUPPORTED_OP"
, "NONFUSABLE_SPEC"
, "NO_INPUTS"
, "NON_FLOATING_INPUT"
, "NO_OUTPUTS"
, "NO_MAP_SIZE"
};

inline std::ostream& operator<<(std::ostream& out, const ReturnCode code) {
  return out << code_strings[code];
}

} // namespace fusers
} // namespace jit
} // namespace torch
