#include "torch/csrc/jit/fusers/common/cache.h"

namespace torch { namespace jit { namespace fusers {

Cache& getCache() {
  static Cache cache;
  return cache;
}

const std::string& Cache::storeOnce(std::shared_ptr<FusionSpec> spec) {
  auto it = cache_map.find(spec->key());
  if (it == cache_map.end()) {
    std::tie(it, std::ignore) = cache_map.emplace(spec->key(), spec);
  }

  return spec->key();
}

const std::string& Cache::storeOnce(Node* fusion_group) {
  return storeOnce(std::make_shared<FusionSpec>(fusion_group));
}

at::optional<std::shared_ptr<FusionSpec>> Cache::get(const std::string& key) {
  auto it = cache_map.find(key);
  if (it == cache_map.end()) {
    return at::nullopt;
  }
  return it->second;
}

} // namespace fusers
} // namespace jit 
} // namespace torch
