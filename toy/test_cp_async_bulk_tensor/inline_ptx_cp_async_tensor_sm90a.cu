// unittest for PTX instruction: cp.async.bulk.tensor (arch: SM90+)

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/numeric/numeric_types.hpp"
#include <cutlass/arch/barrier.h>
#include <cutlass/cuda_host_adapter.hpp>

// Copy a 64x64 tile starting at (row=0,col=0) into shared memory via PTX:
// cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.
//  - gtm: device pointer to TensorMap
extern "C" __global__ void
k_inline_tma_2d_shared_cta(const void *__restrict__ gtm, __half *g_out) {
  extern __shared__ __align__(16) unsigned char smem[];

  // Layout: [tile (64*64*2 bytes)] [mbarrier (8 bytes)]
  half *tile = reinterpret_cast<half *>(smem);
  uint64_t *mbar = reinterpret_cast<uint64_t *>(smem + 64 * 64 * sizeof(half));

  if (threadIdx.x == 0) {
    // One producer thread: expect bytes of the transaction
    cutlass::arch::ClusterTransactionBarrier::init(mbar, 1);
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        mbar, 64 * 64 * sizeof(half));
  }
  __syncthreads();

  //////////////// PTX asm inline ////////////////
  // clang-format off
  /** Syntax of [cp.async.bulk.tensor]
   * // global -> shared::cta
   *
   * cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}{.level::cache_hint}
   *                                    [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo} {, cache-policy}
   *
   * .dst =                  { .shared::cta }
   * .src =                  { .global }
   * .dim =                  { .1d, .2d, .3d, .4d, .5d }
   * .completion_mechanism = { .mbarrier::complete_tx::bytes }
   * .cta_group =            { .cta_group::1, .cta_group::2 }
   * .load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }
   * .level::cache_hint =    { .L2::cache_hint }
  */
  // clang-format on

  // dstMem   -> tile in shared::cta
  // tensorMap-> gtm (generic address in .global)
  // mbar     -> mbar in shared::cta
  // coords   -> {row, col} = {0, 0}
  // cache    -> policy immediate (64-bit). Use 0 for now.
  uint32_t smem_tile = cute::cast_smem_ptr_to_uint(tile);
  uint32_t smem_mbar = cute::cast_smem_ptr_to_uint(mbar);
  uint64_t tensormap = reinterpret_cast<uint64_t>(gtm);
  int32_t tc0 = 0;              // row
  int32_t tc1 = 0;              // col
  uint64_t cache_policy = 0ull; // L2::cache_hint with default policy

  if (threadIdx.x == 0) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global"
                 ".mbarrier::complete_tx::bytes.L2::cache_hint\n\t"
                 " [%0], [%1, {%3, %4}], [%2], %5;\n\t"
                 :
                 : "r"(smem_tile), "l"(tensormap), "r"(smem_mbar), "r"(tc0),
                   "r"(tc1), "l"(cache_policy)
                 : "memory");
  }
  ///////////////////////////////////////////////////////////

  // Wait for completion
  cutlass::arch::ClusterTransactionBarrier::wait(mbar, /*phase=*/0);

  // Copy tile from smem to gmem
  constexpr int num_elements = 64 * 64;
  for (int idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
    g_out[idx] = tile[idx];
  }
}

// Host: Create TensorMap for tensor(M,N) row-major, box=64x64, element=fp16
static CUtensorMap make_tensormap_half_row_major(const void *gmem_addr, int M,
                                                 int N, int ld) {
  CUtensorMap tmap{}; // 128bit, must be aligned to 64B (equivalent to
                      // cute::TmaDescriptor)

  // reference: include/cute/arch/copy_sm90_desc.hpp
  CUtensorMapDataType tma_type =
      cute::TMA::to_CUtensorMapDataType<cute::half_t>();

  constexpr cuuint32_t tma_rank = 2;
  assert((tma_rank >= 2) && "This function only supports `tensor rank >= 2`");

  assert((reinterpret_cast<uint64_t>(gmem_addr) & 0b1111) ==
         0); // Address must be 16B-aligned

  // shape (M,N)
  cuuint64_t gmem_dim_shape[tma_rank] = {static_cast<cuuint64_t>(N),
                                         static_cast<cuuint64_t>(M)};

  // stride in elements, with stride[0] implicitly 1 per PTX doc
  // Fix: referring to CUTLASS, stride[0] will get ignored, but stride[1] must
  // align with 16B.
  cuuint64_t gmem_prob_stride[tma_rank - 1] = {static_cast<cuuint64_t>(ld)};

  for (int i = 0; i < tma_rank - 1; ++i) {
    gmem_prob_stride[i] *= sizeof(cute::half_t);
    assert((gmem_prob_stride[i] & 0b1111) == 0 && "Stride must be 16B aligned");
  }

  // boxDim (tile in SMEM)
  cuuint32_t smem_box_shape[tma_rank] = {64u, 64u};

  // Box stride in elements (row-major contiguous), range: [1,8] !!!
  // reference:
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
  cuuint32_t smem_box_stride[tma_rank] = {1u, 1u};

  CUtensorMapInterleave tma_interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  CUtensorMapSwizzle smem_swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  CUtensorMapL2promotion tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
  CUtensorMapFloatOOBfill tma_oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  // clang-format off
  CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      /* tensorMap */       &tmap, 
      /* tensorDataType */  tma_type, 
      /* tensorRank */      tma_rank, 
      /* globalAddress */   const_cast<void*>(gmem_addr), 
      /* globalDim */       gmem_dim_shape, 
      /* globalStrides */   gmem_prob_stride, 
      /* boxDim */          smem_box_shape,
      /* elementStrides */  smem_box_stride, 
      /* interleave */      tma_interleave, 
      /* swizzle */         smem_swizzle, 
      /* l2Promotion */     tma_l2Promotion, 
      /* oobFill */         tma_oobFill);
  if (res != CUDA_SUCCESS) {
    std::cerr << "Error: Failed to initialize the TMA descriptor, error code: " << res << std::endl;
    std::cerr << "TMA Desc Addr:   " << &tmap
              << "\ntensorDataType " << tma_type
              << "\ntensorRank     " << tma_rank
              << "\ngmem_address   " << const_cast<void*>(gmem_addr)
              << "\nglobalDim      " << *gmem_dim_shape
              << "\nglobalStrides  " << *gmem_prob_stride
              << "\nboxDim         " << *smem_box_shape
              << "\nelementStrides " << *smem_box_stride
              << "\ninterleave     " << tma_interleave
              << "\nswizzle        " << smem_swizzle
              << "\nl2Promotion    " << tma_l2Promotion
              << "\noobFill        " << tma_oobFill << std::endl;
    // clang-format on
    assert(false);
  }
  return tmap;
}

int main() {
  // Initialize params
  constexpr int M = 256, N = 256;
  constexpr int TILE_M = 64, TILE_N = 64;

  size_t bytes = size_t(M) * N * sizeof(__half); // fp16 * M * N
  __half *dA = nullptr;
  cudaMalloc(&dA, bytes);

  __half *d_out = nullptr;
  cudaMalloc(&d_out, TILE_M * TILE_N * sizeof(__half));

  std::vector<__half> hA(M * N);
  for (int i = 0; i < M * N; ++i)
    hA[i] = __float2half(float(i));

  cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);

  // Create TensorMap on host, upload it to device memory
  CUtensorMap tmap = make_tensormap_half_row_major(dA, M, N, N);
  void *d_tmap = nullptr;
  cudaMalloc(&d_tmap, sizeof(CUtensorMap));
  cudaMemcpy(d_tmap, &tmap, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  // Launch: block 64 threads, smem = 64*64*2 + 8
  dim3 grid(1), block(64);
  size_t smem = 64 * 64 * sizeof(__half) + sizeof(uint64_t);
  cudaFuncSetAttribute(k_inline_tma_2d_shared_cta,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  k_inline_tma_2d_shared_cta<<<grid, block, smem>>>(d_tmap, d_out);
  cudaDeviceSynchronize();

  std::vector<__half> h_out(TILE_M * TILE_N);
  cudaMemcpy(h_out.data(), d_out, TILE_M * TILE_N * sizeof(__half),
             cudaMemcpyDeviceToHost);

  // Verify with golden
  int errors = 0;
  for (int m = 0; m < TILE_M; ++m) {
    for (int n = 0; n < TILE_N; ++n) {
      __half v = h_out[m * TILE_N + n];
      __half golden = hA[m * N + n];
      if (abs(__half2float(v) - __half2float(golden)) > 0.00001f) {
        ++errors;
        printf("Error[%d]: golden[%d, %d]=%f, value[%d, %d]=%f\n", errors, m, n,
               __half2float(golden), m, n, __half2float(v));
      }
    }
  }

  if (!errors)
    std::cout << "Test PASS!" << std::endl;
  else
    std::cout << "Test FAIL! Errors: " << errors << std::endl;

  cudaFree(d_tmap);
  cudaFree(dA);
  cudaFree(d_out);
  return 0;
}
