// Minimal inline-PTX demo of cp.async.bulk.tensor (SM90+/SM100)
// Focus: show exact operands mapping without CUTLASS/CUTE wrappers.

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include <cutlass/arch/barrier.h>

// Copy a 64x64 tile starting at (row=0,col=0) into shared memory via PTX:
// cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.
//  - gtm: device pointer to TensorMap
extern "C" __global__ void
k_inline_tma_2d_shared_cta(const void *__restrict__ gtm) {
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



//////////////// PTX operands mapping ////////////////

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

  if (threadIdx.x < 4) {
    float v = __half2float(tile[threadIdx.x]);
    printf("inline-ptx s[0+%u]=%f\n", threadIdx.x, v);
  }
}

// Host: Crete TensorMap for (M,N) row-major, box=64x64, element=fp16
static CUtensorMap make_tensormap_half_row_major(const void *base, int M, int N,
                                                 int ld) {
  CUtensorMap tmap{};
  CUtensorMapDataType type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  uint32_t dim = 2;
  CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  CUtensorMapL2promotion l2promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
  
  // shape (M,N)
  uint64_t shape[2] = {static_cast<uint64_t>(M), static_cast<uint64_t>(N)};
  
  // stride in elements, with stride[0] implicitly 1 per PTX doc
  uint64_t stride_elems[2] = {0,
                              static_cast<uint64_t>(ld)}; // stride[0] ignored
  
  // Box shape (tile in SMEM)
  uint32_t box[2] = {64u, 64u};

  // Box stride in elements (row-major contiguous)
  uint32_t bstride[2] = {64u, 1u};
  CUtensorMapSwizzle swz = CU_TENSOR_MAP_SWIZZLE_64B;

  CUresult res = cuTensorMapEncodeTiled(
      &tmap, type, dim, const_cast<void *>(base), shape, stride_elems + 1, box,
      bstride, interleave, swz, l2promotion, oob);
  if (res != CUDA_SUCCESS) {
    std::cerr << "cuTensorMapEncodeTiled failed: " << res << std::endl;
    std::abort();
  }
  return tmap;
}

int main() {
  // Initialize params
  constexpr int M = 256, N = 256;
  size_t bytes = size_t(M) * N * sizeof(__half); // fp16 * M * N
  __half *dA = nullptr;
  cudaMalloc(&dA, bytes);

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
  k_inline_tma_2d_shared_cta<<<grid, block, smem>>>(d_tmap);
  cudaDeviceSynchronize();

  cudaFree(d_tmap);
  cudaFree(dA);
  return 0;
}
