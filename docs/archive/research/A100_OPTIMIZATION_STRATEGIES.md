# A100 Optimization Strategies

**Purpose:** Complete optimization grammar for A100 (sm_80) CUDA kernel development.

**Target GPU:** NVIDIA A100-SXM4-80GB

---

## Part 1: A100 Hardware Specifications

### 1.1 Compute Resources

| Resource | Value | Optimization Impact |
|----------|-------|---------------------|
| Compute Capability | sm_80 | Determines available instructions |
| SMs (Streaming Multiprocessors) | 108 | Max concurrent blocks |
| CUDA Cores | 6,912 | Parallel throughput |
| Tensor Cores | 432 | Matrix operations |
| Clock Speed | 1.41 GHz | Baseline compute rate |

### 1.2 Memory Hierarchy

| Level | Size | Bandwidth | Latency | Optimization |
|-------|------|-----------|---------|--------------|
| Registers | 65,536 per SM | — | 1 cycle | Maximize utilization |
| L1 Cache | 192 KB per SM | — | ~30 cycles | Data reuse |
| Shared Memory | 164 KB per SM | — | ~30 cycles | Software-managed |
| L2 Cache | 40 MB | — | ~200 cycles | Persistence, pinning |
| HBM2e | 80 GB | 2,039 GB/s | ~400 cycles | Coalescing, prefetch |

### 1.3 Key A100 Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| Async Copy | `cp.async` instruction | Hide memory latency |
| Tensor Cores | TF32, BF16, FP16 | Matrix operations |
| L2 Persistence | Cache pinning API | Repeated access patterns |
| Cooperative Groups | Grid-wide sync | Single-kernel algorithms |
| Warp Primitives | `__shfl_sync`, `__ballot_sync` | Efficient communication |

---

## Part 2: Memory Optimization Grammar

### 2.1 Coalesced Memory Access

**Pattern:** Adjacent threads access adjacent memory locations

```cuda
// GOOD: Coalesced access
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];  // Adjacent threads → adjacent memory
    }
}

// BAD: Strided access
__global__ void bad_access(float* matrix, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    for (int col = 0; col < cols; col++) {
        matrix[row * cols + col] *= 2;  // Strided by cols
    }
}

// GOOD: Transposed for coalescing
__global__ void good_access(float* matrix, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int row = 0; row < rows; row++) {
        matrix[row * cols + col] *= 2;  // Coalesced
    }
}
```

### 2.2 Shared Memory Tiling

**Pattern:** Load tiles into shared memory for reuse

```cuda
__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < N / TILE; t++) {
        // Cooperative load into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();
        
        // Compute on shared memory
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

### 2.3 L2 Cache Persistence

**Pattern:** Pin frequently-accessed data to L2 cache

```cuda
// Setup: Pin parent array for Union-Find
void setup_l2_persistence(int* parent, size_t size) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // A100 has 40MB L2, use 30MB for persistence
    size_t persist_size = 30 * 1024 * 1024;
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_size);
    
    // Configure access policy
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr = parent;
    stream_attribute.accessPolicyWindow.num_bytes = size;
    stream_attribute.accessPolicyWindow.hit_ratio = 1.0;  // Always persist
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
}
```

### 2.4 Async Copy (cp.async)

**Pattern:** Overlap computation with memory transfer

```cuda
__global__ void async_copy_example(float* src, float* dst, int n) {
    __shared__ float buffer[TILE];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Async copy from global to shared
    // Threads cooperatively load a tile
    for (int i = 0; i < TILE; i += blockDim.x) {
        int idx = tid + i;
        if (idx < n) {
            // Async copy - doesn't block
            asm("cp.async.ca.shared.global [%0], [%1], %2;"
                : : "r"(buffer + i), "l"(src + idx), "n"(4));
        }
    }
    
    // Commit and wait
    asm("cp.async.commit;");
    asm("cp.async.wait_all;");
    
    // Now buffer is ready, compute
    // ...
}
```

---

## Part 3: Compute Optimization Grammar

### 3.1 Warp Primitives

**Pattern:** Efficient intra-warp communication

```cuda
// Warp-level reduction
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level broadcast
__device__ float warp_broadcast(float val, int src_lane) {
    return __shfl_sync(0xffffffff, val, src_lane);
}

// Warp-level ballot (bitmask of predicates)
__device__ unsigned int warp_ballot(int predicate) {
    return __ballot_sync(0xffffffff, predicate);
}

// Warp-level all-reduce
__device__ float warp_allreduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    // Broadcast result to all lanes
    return __shfl_sync(0xffffffff, val, 0);
}
```

### 3.2 Register Blocking

**Pattern:** Use registers to avoid repeated memory access

```cuda
__global__ void register_blocking(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Register block: 4x4 output per thread
    float Creg[4][4] = {0};
    float Areg[4];
    float Breg[4];
    
    for (int k = 0; k < N; k += 4) {
        // Load into registers
        for (int i = 0; i < 4; i++) {
            Areg[i] = A[(row * 4 + i) * N + k];
            Breg[i] = B[k * N + col * 4 + i];
        }
        
        // Compute in registers
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int p = 0; p < 4; p++) {
                    Creg[i][j] += Areg[p] * Breg[p];
                }
            }
        }
    }
    
    // Store results
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            C[(row * 4 + i) * N + col * 4 + j] = Creg[i][j];
        }
    }
}
```

### 3.3 Vectorized Loads

**Pattern:** Use float4/int4 for 128-bit memory transactions

```cuda
__global__ void vectorized_load(float4* src, float4* dst, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Single 128-bit load instead of 4 × 32-bit
        float4 data = src[tid];
        
        // Process 4 elements
        data.x *= 2;
        data.y *= 2;
        data.z *= 2;
        data.w *= 2;
        
        dst[tid] = data;
    }
}

// For graph algorithms with CSR format
__global__ void csr_vectorized(int* row_ptr, int* col_idx, float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - 1) {
        int start = row_ptr[tid];
        int end = row_ptr[tid + 1];
        int degree = end - start;
        
        // Vectorized load for high-degree vertices
        if (degree >= 4) {
            for (int i = start; i < end - 3; i += 4) {
                float4 vals = reinterpret_cast<float4*>(data)[i / 4];
                // Process 4 edges at once
            }
        }
    }
}
```

---

## Part 4: Thread Configuration Grammar

### 4.1 Occupancy Maximization

**Pattern:** Choose block size to maximize occupancy

```cuda
// A100: 108 SMs, 2048 threads per SM max
// Optimal block sizes: 128, 256, 512

// For compute-bound kernels
#define BLOCK_SIZE 256  // Good balance

// For memory-bound kernels
#define BLOCK_SIZE 512  // More threads to hide latency

// Launch configuration
int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
kernel<<<blocks, BLOCK_SIZE>>>(args);

// 2D configuration for matrices
dim3 block(16, 16);  // 256 threads
dim3 grid((cols + 15) / 16, (rows + 15) / 16);
matmul_kernel<<<grid, block>>>(args);
```

### 4.2 Register Pressure Management

**Pattern:** Control register usage with launch bounds

```cuda
// Limit registers per thread to increase occupancy
__global__ void __launch_bounds__(256, 4) kernel_with_bounds(...) {
    // 256 threads per block, min 4 blocks per SM
    // Compiler will limit register usage
}

// For register-heavy kernels
__global__ void __launch_bounds__(128, 2) register_heavy_kernel(...) {
    // 128 threads, min 2 blocks per SM
    // Allows more registers per thread
}
```

### 4.3 Cooperative Kernel Launch

**Pattern:** Grid-wide synchronization for single-kernel algorithms

```cuda
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void __launch_bounds__(256) cooperative_wcc(
    int* parent, int* row_ptr, int* col_idx, int n
) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Initialize
    if (tid < n) {
        parent[tid] = tid;
    }
    grid.sync();  // All blocks synchronize
    
    // Phase 2: Iterate until convergence
    bool changed = true;
    while (changed) {
        changed = false;
        
        if (tid < n) {
            int root = find_root(parent, tid);
            for (int e = row_ptr[tid]; e < row_ptr[tid + 1]; e++) {
                int u = col_idx[e];
                int root_u = find_root(parent, u);
                if (root_u != root) {
                    parent[max(root, root_u)] = min(root, root_u);
                    changed = true;
                }
            }
        }
        grid.sync();
        
        // Reduce changed across grid
        // ...
    }
}

// Launch with cooperative groups
void launch_cooperative(cudaStream_t stream) {
    int blocks = 108;  // One per SM for efficiency
    void* args[] = {&parent, &row_ptr, &col_idx, &n};
    cudaLaunchCooperativeKernel((void*)cooperative_wcc, blocks, 256, args, 0, stream);
}
```

---

## Part 5: Algorithm-Specific Patterns

### 5.1 BFS Direction-Optimizing

**Pattern:** Switch between Top-Down and Bottom-Up based on frontier size

```cuda
// Thresholds from WarpSpeed SKILLS.md
#define TD_TO_BU(N) ((N) / 20)    // frontier > 5% of vertices
#define BU_TO_TD(N) ((N) / 200)   // frontier < 0.5% of vertices

__global__ void bfs_top_down(int* row_ptr, int* col_idx, int* frontier, 
                              int* next_frontier, int* distances, 
                              int n, int* frontier_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *frontier_size) return;
    
    int v = frontier[tid];
    for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
        int u = col_idx[e];
        if (distances[u] == -1) {
            distances[u] = distances[v] + 1;
            int idx = atomicAdd(frontier_size, 1);
            next_frontier[idx] = u;
        }
    }
}

__global__ void bfs_bottom_up(int* row_ptr, int* col_idx, int* frontier,
                               int* next_frontier, int* distances,
                               int n, int* frontier_size) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n || distances[u] != -1) return;
    
    for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
        int v = col_idx[e];
        if (distances[v] != -1) {  // v is in frontier
            distances[u] = distances[v] + 1;
            int idx = atomicAdd(frontier_size, 1);
            next_frontier[idx] = u;
            break;  // Found parent, done
        }
    }
}

void bfs_hybrid(int* row_ptr, int* col_idx, int* distances, int n, int source) {
    int frontier_size = 1;
    int level = 0;
    
    while (frontier_size > 0) {
        if (frontier_size > TD_TO_BU(n)) {
            // Large frontier: Bottom-Up is faster
            bfs_bottom_up<<<blocks, threads>>>(...);
        } else {
            // Small frontier: Top-Down is faster
            bfs_top_down<<<blocks, threads>>>(...);
        }
        
        // Check if should switch back
        if (frontier_size < BU_TO_TD(n)) {
            // Switch to Top-Down for next iteration
        }
        
        level++;
    }
}
```

### 5.2 Union-Find for WCC

**Pattern:** Non-atomic Union-Find with path compression

```cuda
__device__ int find_root(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];  // Path compression
        x = parent[x];
    }
    return x;
}

__global__ void wcc_union_find(int* parent, int* row_ptr, int* col_idx, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    int root_v = find_root(parent, tid);
    
    for (int e = row_ptr[tid]; e < row_ptr[tid + 1]; e++) {
        int u = col_idx[e];
        int root_u = find_root(parent, u);
        
        if (root_u != root_v) {
            // Non-atomic union: deterministic ordering
            int lo = min(root_u, root_v);
            int hi = max(root_u, root_v);
            parent[hi] = lo;
            root_v = lo;
        }
    }
}

// Multiple iterations for convergence
void wcc_solve(int* parent, int* row_ptr, int* col_idx, int n) {
    // Initialize
    for (int i = 0; i < n; i++) parent[i] = i;
    
    // Iterate until convergence
    for (int iter = 0; iter < MAX_ITER; iter++) {
        wcc_union_find<<<blocks, threads>>>(parent, row_ptr, col_idx, n);
        cudaDeviceSynchronize();
        
        // Check convergence (optional)
        if (check_convergence(parent, n)) break;
    }
    
    // Finalize: compress all paths
    finalize_labels<<<blocks, threads>>>(parent, n);
}
```

### 5.3 Louvain 3-Tier Dispatch

**Pattern:** Choose parallelization based on graph characteristics

```cuda
// Tier 1: Serial for tiny graphs
void louvain_serial(int* row_ptr, int* col_idx, float* weights, 
                     int* communities, int n) {
    if (n > 200) return;  // Only for N <= 200
    
    // Single-threaded processing
    for (int v = 0; v < n; v++) {
        // Compute delta-Q for each neighbor community
        // Move to best community
    }
}

// Tier 2: Thread-level for low-degree graphs
__global__ void louvain_thread_tier(int* row_ptr, int* col_idx, float* weights,
                                     int* communities, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    
    float best_delta_q = 0.0f;
    int best_community = communities[v];
    
    // Each thread processes one vertex
    for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
        int u = col_idx[e];
        float w = weights[e];
        int c_u = communities[u];
        
        float delta_q = compute_delta_q(v, c_u, w, ...);
        if (delta_q > best_delta_q) {
            best_delta_q = delta_q;
            best_community = c_u;
        }
    }
    
    // Atomic update for community membership
    if (best_community != communities[v]) {
        atomicExch(&communities[v], best_community);
    }
}

// Tier 3: Warp-level for high-degree graphs
__global__ void __launch_bounds__(256) louvain_warp_tier(
    int* row_ptr, int* col_idx, float* weights, int* communities, int n
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    
    if (warp_id >= n) return;
    
    int v = warp_id;
    int start = row_ptr[v];
    int end = row_ptr[v + 1];
    int degree = end - start;
    
    // Warp cooperates on edges
    float best_delta_q = 0.0f;
    int best_community = communities[v];
    
    for (int e = start + lane_id; e < end; e += 32) {
        int u = col_idx[e];
        float w = weights[e];
        int c_u = communities[u];
        
        float delta_q = compute_delta_q(v, c_u, w, ...);
        
        // Warp-level reduce to find best
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, delta_q, offset);
            int other_c = __shfl_down_sync(0xffffffff, c_u, offset);
            if (other > delta_q) {
                delta_q = other;
                c_u = other_c;
            }
        }
        
        if (lane_id == 0 && delta_q > best_delta_q) {
            best_delta_q = delta_q;
            best_community = c_u;
        }
    }
    
    if (lane_id == 0 && best_community != communities[v]) {
        atomicExch(&communities[v], best_community);
    }
}

// Dispatch function
void louvain_dispatch(int* row_ptr, int* col_idx, float* weights,
                       int* communities, int n) {
    if (n <= 200) {
        louvain_serial(row_ptr, col_idx, weights, communities, n);
    } else {
        float avg_degree = (float)(row_ptr[n]) / n;
        
        if (avg_degree < 8) {
            // Thread-level
            louvain_thread_tier<<<blocks, threads>>>(...);
        } else {
            // Warp-level
            louvain_warp_tier<<<blocks, threads>>>(...);
        }
    }
}
```

---

## Part 6: Compilation Flag Grammar

### 6.1 Allowed Flags

| Flag | Purpose | When to Use |
|------|---------|-------------|
| `--use_fast_math` | Fast approximations | Non-critical precision |
| `--extra-device-vectorization` | Vectorize more | Memory-bound kernels |
| `--rdc=true` | Relocatable device code | Cooperative kernels |
| `--maxrregcount=N` | Limit registers | Increase occupancy |

### 6.2 maxrregcount Guidelines

```cuda
// Low register count → high occupancy
// --maxrregcount=32  → More threads per SM, but may spill

// High register count → better performance per thread
// --maxrregcount=128 → Fewer threads, but no spilling

// A100: 65,536 registers per SM
// With 256 threads: 256 registers available per thread
// With --maxrregcount=64: Can run 1024 threads per SM (4 blocks)
```

### 6.3 Architecture Flag

```bash
# A100 (sm_80)
nvcc -arch=sm_80 kernel.cu -o kernel.so --shared

# H100 (sm_90) - for training GPU
nvcc -arch=sm_90 kernel.cu -o kernel.so --shared

# Cross-compile for A100 on H100
nvcc -arch=sm_80 kernel.cu -o kernel.so --shared
```

---

## Part 7: Performance Analysis

### 7.1 Nsight Compute Metrics

| Metric | Target | Optimization |
|--------|--------|--------------|
| `sm__warps_active.avg.pct_of_peak` | > 60% | Occupancy |
| `smsp__sass_thread_inst_executed_op_fp32_pred_on` | High | FLOPs |
| `l1tex__t_sectors_pipe_lsu_mem_ld_op_ld.sum` | Low | Coalescing |
| `lts__t_sectors_op_read.sum` | Low | L2 efficiency |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | < 80% | Not memory-bound |

### 7.2 Profiling Commands

```bash
# Detailed kernel analysis
ncu --set full -o profile ./kernel_exe

# Focus on specific metrics
ncu --metrics sm__warps_active.avg.pct_of_peak ./kernel_exe

# Memory throughput analysis
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_ld_op_ld.sum ./kernel_exe
```

---

## Part 8: Anti-Patterns to Avoid

### 8.1 Memory Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Strided access | Poor coalescing | Transpose data |
| Random access | Cache misses | Sort or bin |
| Excessive atomics | Serialization | Use warp primitives |
| Bank conflicts | Shared memory stalls | Pad shared memory |

### 8.2 Compute Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Divergent warps | Serialized execution | Reorganize threads |
| Low occupancy | Underutilized SMs | Adjust block size |
| Register spilling | Local memory fallback | Limit registers |
| Excessive sync | Barrier overhead | Reduce sync points |

---

## Part 9: Quick Reference

### 9.1 A100 Constants

```cuda
#define A100_SMS 108
#define A100_L2_CACHE (40 * 1024 * 1024)
#define A100_SMEM_PER_SM (164 * 1024)
#define A100_REGS_PER_SM 65536
#define A100_MAX_THREADS_PER_SM 2048
#define A100_MAX_THREADS_PER_BLOCK 1024
#define A100_WARP_SIZE 32
```

### 9.2 Optimization Checklist

- [ ] Coalesced memory access
- [ ] Shared memory tiling for reuse
- [ ] L2 cache persistence for hot data
- [ ] Warp primitives for communication
- [ ] Vectorized loads (float4)
- [ ] Register blocking
- [ ] Appropriate block size (128, 256, 512)
- [ ] Launch bounds for register control
- [ ] Minimal synchronization
- [ ] No bank conflicts

---

## References

| Source | Content |
|--------|---------|
| NVIDIA A100 Whitepaper | Hardware specifications |
| WarpSpeed SKILLS.md | BFS thresholds, Louvain tiers |
| CUDA Agent Paper | Reward function, training pipeline |
| Nsight Compute Docs | Profiling metrics |
