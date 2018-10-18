#include "torch/csrc/jit/fusers/cache.h"

#include <unordered_map>
#include <mutex>
#include <cstdint>

namespace torch { namespace jit { namespace fusers {

static int64_t fusion_counter{0};
static std::unordered_map<int64_t, FusionSpec> specMap_;
static std::mutex mutex_;

int64_t store(std::shared_ptr<Graph> graph) {
  std::lock_guard<std::mutex> guard{mutex_};
  const auto key = fusion_counter++;

  specMap_.emplace(
    std::piecewise_construct
  , std::forward_as_tuple(key)
  , std::forward_as_tuple(key, graph));

  return key;
}

at::optional<FusionSpec&> retrieve(const int64_t key) { 
  std::lock_guard<std::mutex> guard{mutex_};
  auto it = specMap_.find(key);
  if (it == specMap_.end()) return at::nullopt;
  return it->second;
}

} // namespace fusers
} // namespace jit
} // namespace torch
