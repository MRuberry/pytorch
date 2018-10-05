#pragma once
#include "torch/csrc/jit/fusers/config.h"
#if USE_CPU_FUSER || USE_CUDA_FUSER

#include "torch/csrc/utils/disallow_copy.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fusers/common/fusion_spec.h"

#include "ATen/core/optional.h"

#include <string>
#include <unordered_map> 
#include <memory>

namespace torch { namespace jit { namespace fusers {

struct Cache {
  TH_DISALLOW_COPY_AND_ASSIGN(Cache);

  Cache() = default;
  ~Cache() = default;

  // Ensures the given fusion spec is stored in the cache.
  // Returns true if spec was stored, false otherwise.
  bool storeOnce(std::shared_ptr<FusionSpec> spec);

  at::optional<std::shared_ptr<FusionSpec>> get(const std::string& key);

private:
  std::unordered_map<std::string, std::shared_ptr<FusionSpec>> cache_map;
};

Cache& getCache();

} // namespace fusers
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
