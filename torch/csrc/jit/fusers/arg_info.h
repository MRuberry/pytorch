#pragma once

#include "ATen/core/ScalarType.h"
#include "ATen/core/ArrayRef.h"

#include <cstdint>
#include <map>
#include <vector>
#include <set>
#include <iostream>

namespace torch { namespace jit { namespace fusers {

struct ArgDesc {
  ArgDesc() = default;
  ArgDesc(
    const int64_t _name
  , const at::ScalarType _scalar_type
  , const int64_t _numel
  , std::vector<int64_t> _sizes) 
  : name_{_name}
  , scalar_type_{_scalar_type}
  , numel_{_numel}
  , sizes_{_sizes} { }
  ~ArgDesc() = default;

  int64_t name() const { return name_; }
  at::ScalarType scalarType() const { return scalar_type_; }
  int64_t numel() const { return numel_; }
  const std::vector<int64_t>& sizes() const { return sizes_; }
  std::vector<int64_t>& sizes() { return sizes_; } 

private:
  int64_t name_;
  at::ScalarType scalar_type_;
  int64_t numel_;
  std::vector<int64_t> sizes_;
};

struct ArgInfo {
  std::map<int64_t, ArgDesc> arg_map;
  std::set<int64_t> inputs;
  std::set<int64_t> outputs; 
};

inline std::ostream& operator<<(std::ostream& out, const ArgDesc& desc) {
  out << desc.name() << ":" << desc.scalarType() << ":" << desc.numel(); 
  return out;
}


} // namespace fusers
} // namespace jit
} // namespace torch
