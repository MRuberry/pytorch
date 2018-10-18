#include "torch/csrc/jit/fusers/type_and_shape_propagation.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include <iostream> // TODO: remove me

namespace torch { namespace jit { namespace fusers {


// static at::optional<std::vector<int64_t>> getMapSize(
//   FusionSpec& spec
// , at::TensorList args
// , at::IntList arg_subset) {
//   int64_t dim_after_broadcast = 0;
//   for (int64_t arg_idx : arg_subset) {
//     dim_after_broadcast = std::max(dim_after_broadcast, args[arg_idx].dim());
//   }
//   // TODO: this keeps reallocating map_size at every iteration, but we know
//   // exactly how much storage do we need, so this could be fixed in-place at
//   // every step. We're just missing a few functions for ATen, but the fix
//   // should be straightforward.
//   // NB: we leave this uninitialized, because an empty size is trivially
//   // broadcastable to any other size.
//   std::vector<int64_t> map_size;
//   for (size_t i = 0; i < arg_subset.size(); ++i) {
//     auto& arg = args[arg_subset[i]];
//     auto& chunk_desc = spec.inputChunkDescriptors()[arg_subset[i]];
//     if (chunk_desc.nSubTensors() == 1) {
//       try {
//         map_size = at::infer_size(map_size, arg.sizes());
//       } catch (std::exception& e) {
//         return at::nullopt;
//       }
//     } else {
//       auto tensor_sizes = arg.sizes().vec();
//       int64_t num_chunks = chunk_desc.nSubTensors();
//       int64_t dim = at::maybe_wrap_dim(chunk_desc.dim(), tensor_sizes.size());
//       if (tensor_sizes[dim] % num_chunks != 0) {
//         return at::nullopt;
//       }
//       tensor_sizes[dim] /= num_chunks;
//       try {
//         map_size = at::infer_size(map_size, tensor_sizes);
//       } catch (std::exception& e) {
//         return at::nullopt;
//       }
//     }
//   }

//   return {map_size};
// }

// 
// static at::optional<std::vector<int64_t>> canRunKernel(
//   FusionSpec& spec
// , at::TensorList args) {
//   // Short-circuits if # of arguments mismatch
//   AT_CHECK(
//     args.size() == spec.inputChunkDescriptors().size()
//   , "Expected ", spec.inputChunkDescriptors().size(), " arguments, but got ", args.size());


//   at::optional<std::vector<int64_t>> map_size;
//   for (const auto& broadcast_group : spec.inputBroadcastGroups()) {
//     if (!map_size) {
//       map_size = getMapSize(spec, args, broadcast_group);
//       if (!map_size) {
//         return at::nullopt;
//       }
//     } else {
//       auto group_map_size = getMapSize(spec, args, broadcast_group);
//       // NB: this checks that group_map_size is defined AND equal to map_size
//       if (map_size != group_map_size) {
//         return at::nullopt;
//       }
//     }
//   }
//   return map_size;
// }

// // Note: modifies its arguments, although map_size is restored to its original value
// static void expandArgs(
//   FusionSpec& spec
// , std::vector<at::Tensor>& args
// , std::vector<int64_t>& map_size) {
//   for (size_t i = 0; i < args.size(); ++i) {
//     auto& arg = args[i];
//     auto& pdesc = spec.inputChunkDescriptors()[i];
//     if (pdesc.nSubTensors() == 1) {
//       if (arg.sizes().equals(map_size)) continue;
//       arg = arg.expand(map_size);
//     } else {
//       map_size.at(pdesc.dim()) *= pdesc.nSubTensors();
//       if (!arg.sizes().equals(map_size)) {
//         arg = arg.expand(map_size);
//       }
//       map_size.at(pdesc.dim()) /= pdesc.nSubTensors();
//     }
//   }
// }

// The place for device independent "runtime" or "type and shape" propagation
// Note: currently has very specific shape checking and determines how to
// expand inputs.
//
// Expanded note:
// This requires that all values, post-input chunks and pre-output concats,
// have the same shape, called the map size. This could be verified with
// actual shape propagation, but the following code infers this faster
// using properties of broadcast and pointwise operations.
//
// In particular, in a DAG, all tensors are expandable to the size of the
// output tensor. Further, expands can be reordered. Ir pre-concat outputs
// have the same shape, then their expansions can be "pushed" to post-chunk
// inputs, and again all intermediates are of the same shape. The expansion
// performed here ensures all intermediate tensors have the same shape.
// static ReturnCode typeAndShapePropagation(
//   std::map<int64_t, int64_t>& tensor_map
// , FusionSpec& spec
// , Stack& stack) {
  // Acquires stack args
  

  // // Short-circuits if no inputs or non-float tensor input
  // if (inputs.size() == 0) return std::make_tuple(ReturnCode::NO_INPUTS, at::nullopt, at::nullopt);
  // for (const auto& input : inputs) {
  //   if (!isFloatingType(input.scalar_type())) {
  //     return std::make_tuple(ReturnCode::NON_FLOATING_INPUT, at::nullopt, at::nullopt);
  //   }
  // }

  // // Short-circuits if no outputs or non-float tensor output
  // if (spec.graph()->outputs().size() == 0) return std::make_tuple(ReturnCode::NO_OUTPUTS, at::nullopt, at::nullopt);

  // auto maybe_map_size = canRunKernel(spec, inputs);
  // if (!maybe_map_size) {
  //   return std::make_tuple(ReturnCode::NO_MAP_SIZE, at::nullopt, at::nullopt);
  // }
  // expandArgs(spec, inputs, *maybe_map_size);
  // FusionArgSpec arg_spec{inputs};

  // return std::make_tuple(ReturnCode::SUCCESS, arg_spec, *maybe_map_size);
  // return ReturnCode::NOT_IMPL;
// }

ReturnCode typeAndShapePropagation(
  ArgInfo& arg_info
, const FusionSpec& spec
, const Stack& stack) { 

  // Acquires inputs from stack
  const auto inputs = fmap(last(stack, spec.nInputs()), [](const IValue& i) {
    return i.toTensor();
  });

  // Short-circuits if no inputs
  if (inputs.size() == 0) return ReturnCode::NO_INPUTS;

  // Adds inputs 
  // Note: the graph inputs (Value) have the names of the inputs
  for (auto i = decltype(inputs.size()){0}; i < inputs.size(); ++i) {
    const auto& graph_input = spec.graph()->inputs()[i];
    const auto name = graph_input->unique();
    const auto& tensor_input = inputs[i];
    std::vector<int64_t> sizes;
    copy(tensor_input.sizes().begin(), tensor_input.sizes().end(), back_inserter(sizes));
    arg_info.arg_map.emplace(
      std::piecewise_construct
    , std::forward_as_tuple(name)
    , std::forward_as_tuple(
        name
      , tensor_input.scalar_type()
      , tensor_input.numel()
      , sizes)
    );
    arg_info.inputs.emplace(name);
  }

  // DEBUGGING
  // prints inputs
  std::cout << "Inputs: " << std::endl;
  for (const auto& name : arg_info.inputs) {
    const auto& arg_desc = arg_info.arg_map[name];
    std::cout << arg_desc << std::endl;
  }

  // Propagates inputs
  for (const auto& node : spec.graph()->nodes()) {
    
  }

  return ReturnCode::NOT_IMPL;
}

} // namespace fusers
} // namespace jit
} // namespace torch
