#include "torch/csrc/jit/fusers/cuda/compiler.h"

#include "THC/THC.h"
#include "ATen/DeviceGuard.h"
#include "ATen/cuda/CUDAContext.h"
#include "torch/csrc/cuda/cuda_check.h"
#include "torch/csrc/jit/resource_guard.h"

#include "nvrtc.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include <vector>
#include <exception>
#include <mutex>
#include <iostream>

namespace torch { namespace jit { namespace fusers { namespace cuda {

static bool checkCUDAVersion(const cudaDeviceProp& prop) {
  if (  (prop.major >= 6 && CUDA_VERSION < 8000) 
     || (prop.major >= 7 && CUDA_VERSION < 9000)) {
    return false;
  }

  return true;
}

bool compileCUDA(
  const int64_t device
, const std::string& name
, const std::string& code) {
  // TODO: testing only, remove
  std::cout << "compileCUDA" << std::endl;
  std::cout << code;

  // Checks device properties
  cudaDeviceProp prop;
  {
    at::DeviceGuard{(int)device};
    TORCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const auto cuda_version_result = checkCUDAVersion(prop);
    if (!cuda_version_result) return false;
  }

  // Creates and compiles the program using nvrtc
  nvrtcProgram program;
  TORCH_NVRTC_CHECK(nvrtcCreateProgram(
    &program
  , code.c_str()
  , nullptr
  , 0
  , nullptr
  , nullptr));
  const std::string compute = "--gpu-architecture=compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
  const std::vector<const char *> args = {"--std=c++11", compute.c_str(), "-default-device"};
  const auto result = nvrtcCompileProgram(program, args.size(), args.data());
  
  // Throws runtime error if NVRTC failed
  if (result == NVRTC_ERROR_COMPILATION) {
    size_t logsize;
    nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    nvrtcGetProgramLog(program, log.data());
    std::stringstream error_stream;
    error_stream << code << std::endl;
    error_stream << log.data();
    throw std::runtime_error(error_stream.str());
  }
  ResourceGuard holdProgram([&] {
    TORCH_NVRTC_CHECK(nvrtcDestroyProgram(&program));
  });

  TORCH_NVRTC_CHECK(result);

  size_t ptx_size;
  TORCH_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
  std::vector<char> ptx(ptx_size);
  TORCH_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
  CUcontext pctx = 0;
  TORCH_CU_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
     std::unique_lock<std::mutex> cudaFreeMutexLock(
     *(THCCachingAllocator_getCudaFreeMutex()));
     cudaFree(0);
  }
  CUmodule module;
  CUfunction function;
  TORCH_CU_CHECK(cuModuleLoadData(&module, ptx.data()));
  TORCH_CU_CHECK(cuModuleGetFunction(&function, module, name.c_str()));

  int maxBlocks;
  TORCH_CU_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxBlocks
  , function
  , 128
  , 0));
  maxBlocks *= prop.multiProcessorCount;

  return true;
}

} // namespace cuda
} // namespace fusers
} // namespace jit
} // namespace torch
