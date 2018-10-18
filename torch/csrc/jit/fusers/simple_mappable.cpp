#include "torch/csrc/jit/fusers/simple_mappable.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/code_template.h"

#include <unordered_map>

namespace torch { namespace jit { namespace fusers {

static const std::unordered_map<NodeKind, std::string> simple_map_ops = {
  // Unary Ops
  {aten::abs, "absf(${0})"}
, {aten::sigmoid, "1.f / (1.f + expf(-${0}))"}
, {aten::relu, "${0} < 0 ? 0.f : ${0} "}
, {aten::log, "logf(${0})"}
, {aten::log10, "log10f(${0})"}
, {aten::log1p, "log1pf(${0})"}
, {aten::log2,  "log2f(${0})"}
, {aten::lgamma, "lgammaf(${0})"}
, {aten::exp, "expf(${0})"}
, {aten::expm1, "expm1f(${0})"}
, {aten::cos, "cosf(${0})"} 
, {aten::acos, "acosf(${0})"}
, {aten::cosh, "coshf(${0})"}
, {aten::sin, "sinf(${0})"}
, {aten::asin, "asinf(${0})"}
, {aten::sinh, "sinhf(${0})"}
, {aten::tan, "tanf(${0})"}
, {aten::atan, "atanf(${0})"}
, {aten::tanh, "tanhf(${0})"}
, {aten::sqrt, "sqrtf(${0})"}
, {aten::rsqrt, "rsqrtf(${0})"}
, {aten::ceil, "ceilf(${0})"}
, {aten::floor, "floorf(${0})"}
, {aten::round, "roundf(${0})"}
, {aten::trunc, "truncf(${0})"}
, {aten::frac, "fracf(${0})"}
, {aten::reciprocal, "reciprocalf(${0})"}
, {aten::neg, "-${0}"}
  
  // Simple Binary Ops
, {aten::atan2, "atan2(${0}, ${1})"}
, {aten::min, "fminf(${0}, ${1})"}
, {aten::max, "fmaxf(${0}, ${1})"}

  // Binary Ops
  // TODO: some of these ops will not get generated because
  // we only work on float inputs/outputs, but they are here to record
  // that they are valid mappable ops once we handle more type

, {aten::__and__, "${0} && ${1}"}
, {aten::__lshift__, "${0} << ${1}"}
, {aten::__or__, "${0} || ${1}"}
, {aten::__rshift__, "${0} >> ${1}"}
, {aten::__xor__, "${0} ^ ${1}"}
, {aten::div, "${0} / ${1}"}
, {aten::eq, "${0} == ${1}"}
, {aten::fmod, "fmodf(${0}, ${1})"}
, {aten::ge, "(${0} >= ${1})"}
, {aten::gt, "${0} > ${1}"}
, {aten::le, "(${0} <= ${1})"}
, {aten::lt, "${0} < ${1}"}
, {aten::type_as, "(${0})"} // Note: everything is implicitly convertible to float
, {aten::mul, "${0} * ${1}"}
, {aten::ne, "${0} != ${1}"}
, {aten::remainder, "remainderf(${0}, ${1})"}
, {aten::pow, "powf(${0}, ${1})"}

  // Experimental
, {aten::add, "${0} + ${2}*${1}"}
, {aten::sub, "(${0} - ${2}*${1})"}
, {aten::rand_like, "uniform(rnd())"}

  // Note: clamp propagates NaN inputs and ignores min or max when they are NaN
, {aten::clamp, "(${0}<${1}?${1}:(${0}>${2}?${2}:${0}))"}

  // Derivatives
, {aten::_sigmoid_backward, "${0} * ${1} * (1.f - ${1})"}
, {aten::_tanh_backward,    "${0} * (1.f - ${1} * ${1})"}
};

static std::string valueName(const Value* const n) {
  return "n" + std::to_string(n->unique());
}

static std::string scalarValue(const int64_t v) {
  return std::to_string(v);
}

// Note: The NAN, NEG_INFINITY and POS_INFINITY strings map to device-specific
// implementations of these special values. These macros are found in the 
// resource strings for each device.
static std::string scalarValue(double v) {
  std::ostringstream out;
  if (std::isnan(v)) {
    out << "NAN";
  } else if (std::isinf(v)) {
    if (v < 0) {
      out << "NEG_INFINITY";
    } else {
      out << "POS_INFINITY";
    }
  } else {
    out << std::scientific << v << "f";
  }
  return out.str();
}

static std::string encodeRHS(const Node* const n) {
  if (n->kind() == prim::Constant) {
    auto val = toIValue(n->output()).value();
    if (val.isDouble()) {
      return scalarValue(val.toDouble());
    } else {
      JIT_ASSERT(val.isInt());
      return scalarValue(val.toInt());
    }
  }

  TemplateEnv env;
  size_t i = 0;
  for(auto in : n->inputs()) {
    env.s(std::to_string(i++), valueName(in));
  }

  const auto& str = simple_map_ops.at(n->kind());
  return format(str, env);
}

bool isSimpleMap(const Node* n) {
  return (simple_map_ops.find(n->kind()) != simple_map_ops.end());
}

} // namespace fusers
} // namespace jit
} // namespace torch
